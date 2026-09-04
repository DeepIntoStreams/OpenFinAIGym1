#!/bin/bash
# SFT-eval on a SLURM cluster with Singularity/Apptainer: merge the LoRA
# adapter into the base, serve the merged model with vLLM, then run the
# zero-shot SingleShot agent (openfinai_skyrl.eval.trained) over the 5
# held-out test tasks via run_trial on the Singularity env.
#
# eval.trained has no --environment-import-path: run_trial picks the
# Singularity env from $HARBOR_TRIAL_CONFIG (same mechanism the no-RL
# baseline uses).
#
# Eval prompt = financial_competitive.txt (the prompt the corpus was prepared
# with) so the measurement is train-matched. The earlier base-model baselines
# used the *diverse* prompt, so they are NOT a like-for-like comparison.
#
# Site-specific knobs (env vars; see scripts/rl/slurm_smoke.sh): CONDA_ROOT,
# CONDA_ENV, PROJECT_DIR, HF_HOME, CUDA_HOME. No SKYRL_DIR needed here.
#
# Usage (one job per size):
#   sbatch --job-name=ofg-sfteval-1p7b --export=ALL,SIZE=1.7b scripts/sft/slurm_sft_eval.sh
#   sbatch --job-name=ofg-sfteval-4b   --export=ALL,SIZE=4b   scripts/sft/slurm_sft_eval.sh
#   sbatch --job-name=ofg-sfteval-8b   --export=ALL,SIZE=8b   scripts/sft/slurm_sft_eval.sh

#SBATCH --job-name=ofg-sfteval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=72

# No `set -e`: we want to always tear down vLLM and dump its log on failure.
set -uo pipefail

SIZE="${SIZE:-1.7b}"
case "${SIZE}" in
    1.7b|4b|8b) : ;;
    *) echo "unknown SIZE: ${SIZE} (use 1.7b|4b|8b)" >&2; exit 1 ;;
esac
SIZE_TAG="${SIZE/./p}"

# Repo root: prefer the sbatch submit dir (usage assumes `sbatch` from the
# repo root) since under sbatch BASH_SOURCE points into SLURM's spool dir.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/data/run_output/experiments-sft/runs/sft_single_qwen3_${SIZE_TAG}}"
MODEL_NAME="${MODEL_NAME:-sft-qwen3-${SIZE_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/eval_test}"
TASKS_ROOT="${TASKS_ROOT:-${PROJECT_DIR}/tasks}"
SYS_PROMPT="${SYS_PROMPT:-examples/prompts/financial_competitive.txt}"

# Eval generation budget. The corpus targets are short code (<~500 tok, no
# <think>), so 8192 is generous + train-matched. --max-model-len = input+output.
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-8192}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
TEMPERATURE="${TEMPERATURE:-0.7}"
N_SEEDS="${N_SEEDS:-3}"          # >1 so success_rate is a real per-task fraction
SEED_BASE="${SEED_BASE:-0}"      # first seed; shard parallel jobs with distinct bases
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

# Conda environment
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

# HF cache. Compute nodes are often network-isolated: the base model must be
# pre-downloaded into HF_HOME for the LoRA merge. Set HF_HUB_OFFLINE=0
# TRANSFORMERS_OFFLINE=0 if the nodes can reach the Hub.
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/data/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# The LoRA merge calls transformers' save_pretrained -> accelerate.unwrap_model,
# which imports deepspeed; deepspeed's import-time op-builder check aborts with
# MissingCUDAException unless CUDA_HOME points at a toolkit. It never compiles
# (the merged model isn't a DeepSpeedEngine) — the import just needs to succeed.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PYTHONUNBUFFERED=1
export TMPDIR="/tmp/ofg_tmp_${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/inductor"
export TRITON_CACHE_DIR="${TMPDIR}/triton"
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

# Per-job port + verifier registry isolation (concurrent eval jobs on a shared
# filesystem).
VLLM_PORT="${VLLM_PORT:-$(( 8000 + (${SLURM_JOB_ID:-0} % 1500) ))}"
export OPENFINAI_VERIFIER_RUN_OUTPUT_ROOT="${PROJECT_DIR}/data/run_output/verifier_runs/${SLURM_JOB_ID:-local}"
mkdir -p "${OPENFINAI_VERIFIER_RUN_OUTPUT_ROOT}"
# run_trial uses the Singularity env from this config (SIF + conda bind).
export HARBOR_TRIAL_CONFIG="${HARBOR_TRIAL_CONFIG:-${PROJECT_DIR}/config/harbor_trial_rl_singularity.yaml}"
# litellm hosted_vllm / openai-compatible: dummy keys keep it happy.
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HOSTED_VLLM_API_KEY="${HOSTED_VLLM_API_KEY:-EMPTY}"

mkdir -p "${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "=== SFT-eval ${SIZE} | run_dir=${RUN_DIR} | model=${MODEL_NAME} | job ${SLURM_JOB_ID:-(none)} on $(hostname) ==="
echo "eval: seeds=${N_SEEDS} seed_base=${SEED_BASE} temp=${TEMPERATURE} max_tokens=${MAX_TOKENS} prompt=${SYS_PROMPT} port=${VLLM_PORT} out=${OUTPUT_DIR}"

# 1. Merge base + LoRA adapter -> <RUN_DIR>/hf_merged (idempotent).
echo "[eval] merging adapter ..."
python -c "from pathlib import Path; from openfinai_skyrl.eval.trained import _auto_merge_adapter; print('MERGED='+str(_auto_merge_adapter(Path('${RUN_DIR}').resolve())))" \
    || { echo "[eval] merge FAILED"; exit 1; }
MERGED_DIR="${RUN_DIR}/hf_merged"

# 2. Serve the merged model with vLLM (OpenAI-compatible) in the background.
echo "[serve] vllm serve ${MERGED_DIR} as ${MODEL_NAME} on :${VLLM_PORT}"
vllm serve "${MERGED_DIR}" \
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
    echo "[serve] vLLM not ready after 900s — tail:"; tail -60 "${OUTPUT_DIR}/vllm_serve.log"; exit 1
fi

# 3. Eval over the 5 held-out test tasks. hosted_vllm needs nested model_info
# (token limits + costs); passed as a JSON --ak (run_trial._parse_kv decodes it).
MODEL_INFO="{\"max_input_tokens\":${MAX_INPUT_TOKENS},\"max_output_tokens\":${MAX_OUTPUT_TOKENS},\"input_cost_per_token\":0,\"output_cost_per_token\":0}"
echo "[eval] eval.trained: 5 test tasks x ${N_SEEDS} seeds via run_trial(Singularity)"
python -m openfinai_skyrl.eval.trained \
    --mode single \
    --checkpoint-dir "${RUN_DIR}" \
    --provider hosted_vllm \
    --model "${MODEL_NAME}" \
    --tasks-root "${TASKS_ROOT}" \
    --test-tasks \
        1_commodity_logreturn_forecasting \
        1_crypto_variant3_logreturn_forecasting \
        1_equity_variant2_logreturn_forecasting \
        1_treasury_variant2_forecasting \
        1_fx_variant4_logreturn_forecasting \
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
