#!/usr/bin/env bash
# Run the SFT pipeline from collection through reporting.
# Stages: collect -> prepare -> eval-baseline -> train -> auto-merge ->
#         eval-trained -> visualize.
# Defaults come from config/sft.yaml (or $CONFIG_PATH).
#
# Usage:
#   bash scripts/sft/run_sft_pipeline.sh                                  # full run
#   MODE=multi bash scripts/sft/run_sft_pipeline.sh
#   SKIP_STAGES=collect,eval-baseline,eval-trained bash scripts/sft/run_sft_pipeline.sh
#   ONLY_STAGES=prepare,train,visualize bash scripts/sft/run_sft_pipeline.sh
#
# Common overrides: MODE, CONFIG_PATH, EXPERIMENT_ROOT, JOBS_DIR, PROVIDER,
# MODEL, SKIP_STAGES, ONLY_STAGES, and N_SEEDS. MANAGE_VLLM=true lets the
# pipeline own the evaluation servers; VLLM_* and CUDA_HOME tune that path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/config/sft.yaml}"
MODE="${MODE:-single}"

CLI=("${PYTHON_BIN}" -m openfinai_skyrl.entrypoints.main_sft_pipeline
     --config "${CONFIG_PATH}" --mode "${MODE}")

append_if_set() {
    local flag="$1"
    local value="${2:-}"
    if [[ -n "${value}" ]]; then
        CLI+=("${flag}" "${value}")
    fi
}

append_if_set --experiment-root "${EXPERIMENT_ROOT:-}"
append_if_set --jobs-dir        "${JOBS_DIR:-}"
append_if_set --tasks-root      "${TASKS_ROOT:-}"
append_if_set --provider        "${PROVIDER:-}"
append_if_set --model           "${MODEL:-}"
append_if_set --agent-import-path "${AGENT_IMPORT_PATH:-}"
append_if_set --n-attempts      "${N_ATTEMPTS:-}"
append_if_set --n-concurrent    "${N_CONCURRENT:-}"
append_if_set --system-prompt-path "${SYSTEM_PROMPT_PATH:-}"
append_if_set --reward-max      "${REWARD_MAX:-}"
append_if_set --top-k-lowest-reward "${TOP_K_LOWEST_REWARD:-}"
append_if_set --model-path      "${MODEL_PATH:-}"
append_if_set --num-steps       "${NUM_STEPS:-}"
append_if_set --max-length      "${MAX_LENGTH:-}"
append_if_set --max-tokens      "${MAX_TOKENS:-}"
append_if_set --n-seeds         "${N_SEEDS:-}"
append_if_set --seed-base       "${SEED_BASE:-}"
append_if_set --vllm-port       "${VLLM_PORT:-}"
append_if_set --vllm-max-model-len "${VLLM_MAX_MODEL_LEN:-}"
append_if_set --skip-stages     "${SKIP_STAGES:-}"
append_if_set --only-stages     "${ONLY_STAGES:-}"

if [[ "${SKIP_ENDPOINT_CHECK:-}" == "true" || "${SKIP_ENDPOINT_CHECK:-}" == "1" ]]; then
    CLI+=(--skip-endpoint-check)
fi

if [[ "${MANAGE_VLLM:-}" == "true" || "${MANAGE_VLLM:-}" == "1" ]]; then
    CLI+=(--manage-vllm)
    # Optionally expose site-specific vLLM and sandbox tooling.
    if [[ -n "${VLLM_TOOLCHAIN_PATH:-}" ]]; then
        export PATH="${VLLM_TOOLCHAIN_PATH}:${PATH}"
    fi
    export PATH="${PYTHON_BIN%/python}:${PATH}"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export CPATH="${CPATH:-${CUDA_HOME}/targets/x86_64-linux/include}"
    export LIBRARY_PATH="${LIBRARY_PATH:-${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/lib64}"
    export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
fi

cd "${PROJECT_DIR}"
echo "[pipeline-wrapper] ${CLI[*]}"
exec "${CLI[@]}"
