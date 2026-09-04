#!/bin/bash
# Eval a trained RL checkpoint on the 5 held-out test tasks (SLURM cluster
# with Singularity/Apptainer).
#
# Differs from slurm_sft_eval.sh: RL exports are already full HF models
# (SkyRL hf_save), so we SKIP the LoRA merge and point `vllm serve` straight at
# the export's policy/ dir. Inference-only single vLLM server on its own port —
# none of the colocated-training eng/port issues apply here.
#
# Prompt defaults to financial_competitive.txt — the canonical RL training
# prompt. Override via SYS_PROMPT for checkpoints trained with another prompt.
#
# Site-specific knobs (env vars; see slurm_smoke.sh): CONDA_ROOT, CONDA_ENV,
# PROJECT_DIR, HF_HOME, CUDA_HOME. CKPT may be absolute or relative to the
# repo root.
#
#   sbatch --job-name=ofg-rleval-1p7b-s100 \
#     --export=ALL,CKPT=data/run_output/rl_runs/<run>/exports/global_step_100/policy,MODEL_NAME=rl-1p7b-b1-s100 \
#     scripts/rl/slurm_rl_eval.sh

#SBATCH --job-name=ofg-rleval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --mem=115G
#SBATCH --cpus-per-gpu=72

set -uo pipefail

CKPT="${CKPT:?set CKPT=<RL export policy dir (full HF model)>}"
MODEL_NAME="${MODEL_NAME:-rl-eval}"
# Repo root: prefer the sbatch submit dir (usage assumes `sbatch` from the
# repo root) since under sbatch BASH_SOURCE points into SLURM's spool dir.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPT}/eval_test}"
TASKS_ROOT="${TASKS_ROOT:-${PROJECT_DIR}/tasks}"
SYS_PROMPT="${SYS_PROMPT:-examples/prompts/financial_competitive.txt}"
# Space-separated held-out test-task list (override via env). Default = the 5
# standard original test tasks; pass the additional OOD set via TEST_TASKS=...
TEST_TASKS="${TEST_TASKS:-1_commodity_logreturn_forecasting 1_crypto_variant3_logreturn_forecasting 1_equity_variant2_logreturn_forecasting 1_treasury_variant2_forecasting 1_fx_variant4_logreturn_forecasting}"

MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-8192}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
TEMPERATURE="${TEMPERATURE:-0.7}"
N_SEEDS="${N_SEEDS:-3}"
SEED_BASE="${SEED_BASE:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniforge3}"
CONDA_ENV="${CONDA_ENV:-openfingym}"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/bin/activate" "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/src:${PYTHONPATH:-}"

for f in src/openfinai_pipeline/__init__.py \
         src/openfinai_pipeline/benchmark/__init__.py \
         src/openfinai_pipeline/corpus/__init__.py; do
    : > "${PROJECT_DIR}/$f"
done

# HF cache (tokenizer/config only — the checkpoint itself is a local dir).
# Set HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 if the nodes can reach the Hub.
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/data/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PYTHONUNBUFFERED=1
export TMPDIR="/tmp/ofg_tmp_${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/inductor"
export TRITON_CACHE_DIR="${TMPDIR}/triton"
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

VLLM_PORT="${VLLM_PORT:-$(( 8000 + (${SLURM_JOB_ID:-0} % 1500) ))}"
export OPENFINAI_VERIFIER_RUN_OUTPUT_ROOT="${PROJECT_DIR}/data/run_output/verifier_runs/${SLURM_JOB_ID:-local}"
mkdir -p "${OPENFINAI_VERIFIER_RUN_OUTPUT_ROOT}"
export HARBOR_TRIAL_CONFIG="${HARBOR_TRIAL_CONFIG:-${PROJECT_DIR}/config/harbor_trial_rl_singularity.yaml}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HOSTED_VLLM_API_KEY="${HOSTED_VLLM_API_KEY:-EMPTY}"

mkdir -p "${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "=== RL-eval | ckpt=${CKPT} | model=${MODEL_NAME} | job ${SLURM_JOB_ID:-(none)} on $(hostname) ==="
echo "eval: seeds=${N_SEEDS} temp=${TEMPERATURE} prompt=${SYS_PROMPT} port=${VLLM_PORT} out=${OUTPUT_DIR}"

# Serve the RL export directly (already a full HF model — no LoRA merge).
echo "[serve] vllm serve ${CKPT} as ${MODEL_NAME} on :${VLLM_PORT}"
vllm serve "${CKPT}" \
    --served-model-name "${MODEL_NAME}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --dtype bfloat16 \
    --enforce-eager \
    > "${OUTPUT_DIR}/vllm_serve.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT

echo "[serve] waiting for /v1/models ..."
ready=0
for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        ready=1; echo "[serve] ready after ~$((i*5))s"; break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "[serve] vLLM exited early — tail of vllm_serve.log:"; tail -60 "${OUTPUT_DIR}/vllm_serve.log"; exit 1
    fi
    sleep 5
done
if [[ "${ready}" != "1" ]]; then
    echo "[serve] vLLM not ready in time — tail:"; tail -60 "${OUTPUT_DIR}/vllm_serve.log"; exit 1
fi

MODEL_INFO="{\"max_input_tokens\":${MAX_INPUT_TOKENS},\"max_output_tokens\":${MAX_OUTPUT_TOKENS},\"input_cost_per_token\":0,\"output_cost_per_token\":0}"
echo "[eval] eval.trained: tasks=[${TEST_TASKS}] x ${N_SEEDS} seeds via run_trial(Singularity)"
python -m openfinai_skyrl.eval.trained \
    --mode single \
    --checkpoint-dir "${CKPT}" \
    --provider hosted_vllm \
    --model "${MODEL_NAME}" \
    --tasks-root "${TASKS_ROOT}" \
    --test-tasks ${TEST_TASKS} \
    --output-dir "${OUTPUT_DIR}" \
    --system-prompt-path "${SYS_PROMPT}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --n-seeds "${N_SEEDS}" \
    --seed-base "${SEED_BASE}" \
    --timeout-sec "${TIMEOUT_SEC}" \
    --endpoint-url "http://127.0.0.1:${VLLM_PORT}" \
    --ak "api_base=http://127.0.0.1:${VLLM_PORT}/v1" \
    --ak "model_info=${MODEL_INFO}"
RC=$?

echo "[eval] eval.trained rc=${RC}"
echo "[eval] === summary.json ==="
cat "${OUTPUT_DIR}/summary.json" 2>/dev/null || echo "(no summary.json)"
exit ${RC}
