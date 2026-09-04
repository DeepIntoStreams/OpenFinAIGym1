#!/usr/bin/env bash
# Run Harbor trials with the host RPC verifier available.
#
#   ./scripts/run_harbor.sh <task_id> [<task_id> ...] [-- <extra args>]
#
# Example:
#   ./scripts/run_harbor.sh offline_stock_forecasting \
#       --model ollama/qwen2.5-coder:32b \
#       --provider ollama \
#       --agent-import-path openfinai_harbor.agents:SingleShotLLMAgent \
#       --turns 5 --temp 0.0
#
# Key flags: --model, --provider, --agent-import-path, --turns, --temp,
# --trials-dir, and --timeout-sec. Arguments after -- pass through unchanged.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Defaults
DEFAULT_MODEL="openrouter/openai/gpt-5.1"
DEFAULT_AGENT_IMPORT_PATH="openfinai_harbor.agents:ReActAgent"
DEFAULT_TURNS=5
DEFAULT_TEMP=0.0
DEFAULT_TRIALS_DIR="jobs/run_harbor"
DEFAULT_TIMEOUT_SEC=600

# Argument parsing
MODEL="$DEFAULT_MODEL"
PROVIDER=""
AGENT_IMPORT_PATH="$DEFAULT_AGENT_IMPORT_PATH"
TURNS="$DEFAULT_TURNS"
TEMP="$DEFAULT_TEMP"
TRIALS_DIR="$DEFAULT_TRIALS_DIR"
TIMEOUT_SEC="$DEFAULT_TIMEOUT_SEC"
TASKS=()
EXTRA_ARGS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model)               MODEL="$2"; shift 2 ;;
    --provider)            PROVIDER="$2"; shift 2 ;;
    --agent-import-path)   AGENT_IMPORT_PATH="$2"; shift 2 ;;
    --turns)               TURNS="$2"; shift 2 ;;
    --temp|--temperature)  TEMP="$2"; shift 2 ;;
    --trials-dir)          TRIALS_DIR="$2"; shift 2 ;;
    --timeout-sec)         TIMEOUT_SEC="$2"; shift 2 ;;
    --)                    shift; EXTRA_ARGS=("$@"); break ;;
    --*)
      echo "Unknown flag: $1" >&2
      exit 2 ;;
    *)
      TASKS+=("$1"); shift ;;
  esac
done

if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "Usage: $0 <task_id> [<task_id> ...] [--model …] [--agent-import-path …] [--turns N] [--temp F]" >&2
  exit 2
fi

# Derive the provider from the model prefix when omitted.
if [[ -z "$PROVIDER" ]]; then
  case "$MODEL" in
    ollama/*)     PROVIDER="ollama" ;;
    openrouter/*) PROVIDER="openrouter" ;;
    anthropic/*)  PROVIDER="anthropic" ;;
    openai/*)     PROVIDER="openai" ;;
    *)            PROVIDER="" ;;
  esac
fi

# Pass provider and bare model separately; run_trial recombines them.
BARE_MODEL="$MODEL"
if [[ -n "$PROVIDER" && "$MODEL" == "${PROVIDER}/"* ]]; then
  BARE_MODEL="${MODEL#${PROVIDER}/}"
fi

# Activate the project environment when present.
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
cd "$ROOT"

echo
echo "Harbor trial runner"
echo "==================="
echo "Tasks:               ${TASKS[*]}"
echo "Model:               $MODEL  (provider=$PROVIDER bare=$BARE_MODEL)"
echo "Agent:               $AGENT_IMPORT_PATH"
echo "Turns:               $TURNS"
echo "Temperature:         $TEMP"
echo "Trials dir:          $TRIALS_DIR"
echo "Timeout per trial:   ${TIMEOUT_SEC}s"
echo

for task_name in "${TASKS[@]}"; do
  task_path="tasks/${task_name}"
  if [[ ! -d "$task_path" ]]; then
    echo "Task not found: ${task_name}" >&2
    exit 1
  fi
done

# Run each task sequentially
for task_name in "${TASKS[@]}"; do
  echo
  echo "=== Running ${task_name} ==="

  AK_ARGS=()
  if [[ -n "$TURNS" ]]; then AK_ARGS+=(--ak "max_turns=${TURNS}"); fi
  if [[ -n "$TEMP" ]];  then AK_ARGS+=(--ak "temperature=${TEMP}"); fi

  PROVIDER_ARGS=()
  if [[ -n "$PROVIDER" ]]; then PROVIDER_ARGS+=(--provider "$PROVIDER"); fi

  python -m openfinai_harbor.run_trial \
    --task-dir "tasks/${task_name}" \
    --agent-import-path "$AGENT_IMPORT_PATH" \
    --model "$BARE_MODEL" \
    "${PROVIDER_ARGS[@]}" \
    --temperature "$TEMP" \
    "${AK_ARGS[@]}" \
    --trials-dir "$TRIALS_DIR" \
    --timeout-sec "$TIMEOUT_SEC" \
    "${EXTRA_ARGS[@]}"
done
