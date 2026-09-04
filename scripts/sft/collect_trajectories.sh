#!/bin/bash
# Collect SFT trajectories via run_trial (which pre-spawns the host verifier
# and forwards VERIFIER_URL/TOKEN/AGENT_ID into the container). Bare `harbor
# run` does not, and crashes on /eval-data/ lookups.
# Reads 1_* forecasting tasks from registry.json; env vars override defaults.

set -uo pipefail   # NOT -e: a single failing trial must not abort the sweep

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

MODE="${MODE:-single}"
if [[ "${MODE}" == "multi" ]]; then
    DEFAULT_AGENT="openfinai_harbor.agents:ReActFinAgent"
else
    DEFAULT_AGENT="openfinai_harbor.agents:SingleShotLLMAgent"
fi

# `claude-cli/<model>` is an optional provider that shells out to a locally installed Claude Code CLI.
MODEL="${MODEL:-openrouter/openai/gpt-5.4-mini}"
AGENT_IMPORT_PATH="${AGENT_IMPORT_PATH:-${DEFAULT_AGENT}}"
# Unset SYSTEM_PROMPT_FILE: BaseFinAgent composes
# DEFAULT_DOMAIN_PROMPT + MODE_PROMPT_ADDENDUM from the agent class.
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-}"
N_ATTEMPTS="${N_ATTEMPTS:-4}"
N_CONCURRENT="${N_CONCURRENT:-2}"
MAX_TURNS="${MAX_TURNS:-10}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"
DESPAWN_GRACE_MIN="${DESPAWN_GRACE_MIN:-15.0}"
TRIALS_DIR="${TRIALS_DIR:-${PROJECT_DIR}/data/run_output/jobs/collect_trajectories}"
REGISTRY_PATH="${REGISTRY_PATH:-${PROJECT_DIR}/registry.json}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

if [[ -n "${SYSTEM_PROMPT_FILE}" && ! -r "${SYSTEM_PROMPT_FILE}" ]]; then
    echo "ERROR: SYSTEM_PROMPT_FILE was set but not readable: ${SYSTEM_PROMPT_FILE}" >&2
    exit 1
fi

if [[ ! -f "${REGISTRY_PATH}" ]]; then
    echo "ERROR: registry.json not found at ${REGISTRY_PATH}" >&2
    echo "Generate one with: python scripts/maintenance/sync_registry.py  (or your project's equivalent)" >&2
    exit 1
fi

# Forwarded into the agent container by runner.build_container_env.
export CLAUDE_CLI_TIMEOUT_SEC="${CLAUDE_CLI_TIMEOUT_SEC:-1800}"
export HARBOR_SUBMIT_TIMEOUT_SEC="${HARBOR_SUBMIT_TIMEOUT_SEC:-600}"

TASKS=$("${PYTHON_BIN}" - <<EOF
import json, sys
with open("${REGISTRY_PATH}") as f:
    reg = json.load(f)
if isinstance(reg, list):
    for pkg in reg:
        for t in pkg.get("tasks", []):
            name = t.get("name", "")
            if name.startswith("1_"):
                print(name)
elif isinstance(reg, dict):
    for t in reg.get("tasks", []):
        name = t.get("name", "")
        if name.startswith("1_"):
            print(name)
EOF
)

if [[ -z "${TASKS}" ]]; then
  echo "[collect_trajectories] no 1_* tasks found in ${REGISTRY_PATH}" >&2
  exit 1
fi

NUM_TASKS=$(echo "${TASKS}" | wc -l | tr -d ' ')
TOTAL_TRIALS=$((NUM_TASKS * N_ATTEMPTS))

echo "=== Collecting trajectories ==="
echo "Registry:        ${REGISTRY_PATH}"
echo "Mode:            ${MODE}"
echo "Model:           ${MODEL}"
echo "Agent:           ${AGENT_IMPORT_PATH}"
echo "System prompt:   ${SYSTEM_PROMPT_FILE:-<in-agent default: DEFAULT_DOMAIN_PROMPT + MODE_PROMPT_ADDENDUM>}"
echo "Tasks:           ${NUM_TASKS}"
echo "Attempts/task:   ${N_ATTEMPTS}    (total trials: ${TOTAL_TRIALS})"
echo "Concurrent/task: ${N_CONCURRENT}"
echo "Max turns:       ${MAX_TURNS}"
echo "Temperature:     ${TEMPERATURE}"
echo "Agent step max:  ${TIMEOUT_SEC}s"
echo "Trials dir:      ${TRIALS_DIR}"
echo "Env:             CLAUDE_CLI_TIMEOUT_SEC=${CLAUDE_CLI_TIMEOUT_SEC} HARBOR_SUBMIT_TIMEOUT_SEC=${HARBOR_SUBMIT_TIMEOUT_SEC}"
echo "Tasks:"
echo "${TASKS}" | sed 's/^/  - /'
echo ""

SUCC=0
FAIL=0
for task in ${TASKS}; do
  TASK_DIR="tasks/${task}"
  if [[ ! -d "${TASK_DIR}" ]]; then
    echo "[collect_trajectories] SKIP: ${TASK_DIR} not on disk"
    FAIL=$((FAIL + N_ATTEMPTS))
    continue
  fi

  echo ""
  echo ">>> ${task}  (${N_ATTEMPTS} attempts, ${N_CONCURRENT}-way) <<<"

  # Concurrent attempts share one per-task verifier (FileLock-guarded spawn).
  COMMON_AK=(--ak "max_turns=${MAX_TURNS}" --ak "temperature=${TEMPERATURE}")
  if [[ -n "${SYSTEM_PROMPT_FILE}" ]]; then
      COMMON_AK+=(--ak "system_prompt_path=${SYSTEM_PROMPT_FILE}")
  fi
  if seq 1 "${N_ATTEMPTS}" | xargs -P "${N_CONCURRENT}" -I {} \
      "${PYTHON_BIN}" -m openfinai_harbor.run_trial \
        --task-dir "${TASK_DIR}" \
        --agent-import-path "${AGENT_IMPORT_PATH}" \
        --model "${MODEL}" \
        --temperature "${TEMPERATURE}" \
        "${COMMON_AK[@]}" \
        --trials-dir "${TRIALS_DIR}" \
        --timeout-sec "${TIMEOUT_SEC}" \
        --despawn-grace-min "${DESPAWN_GRACE_MIN}"; then
    SUCC=$((SUCC + N_ATTEMPTS))
  else
    echo "[collect_trajectories] WARN: at least one attempt for ${task} returned non-zero"
    FAIL=$((FAIL + 1))
    SUCC=$((SUCC + N_ATTEMPTS - 1))
  fi
done

echo ""
echo "=== Collection complete ==="
echo "Trials under ${TRIALS_DIR}/  (timestamped sub-dirs)"
echo "Approximate succ/fail (per task xargs return): ${SUCC} / ${FAIL}"
echo "Next: ${PYTHON_BIN} -m openfinai_skyrl.data.prepare \\"
echo "        --mode ${MODE} \\"
echo "        --jobs-dir ${TRIALS_DIR} \\"
echo "        --tasks-root tasks \\"
echo "        --output-root data/run_output/experiments-sft"
echo "      (prepare reads the system prompt from each trial's trajectory.json"
echo "       — same byte sequence the agent installed at collection time.)"
