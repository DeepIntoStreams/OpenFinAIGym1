#!/bin/bash
# No-RL baseline on a SLURM cluster with Singularity/Apptainer: serve ONE
# model with vLLM, then run the zero-shot SingleShot agent
# (openfinai_skyrl.eval.baseline) over the five held-out forecasting test
# tasks via run_trial on the Singularity env.
#
# One job per model; each writes its own summary.json under
# data/run_output/baselines/<served>-<jobid>/. Aggregate the per-model
# summaries into a table afterwards with scripts/rl/baseline_table.py.
#
# Site-specific knobs (env vars; see slurm_smoke.sh): SKYRL_DIR (required),
# CONDA_ROOT, CONDA_ENV, PROJECT_DIR, HF_HOME, CUDA_HOME.
#
# Override per model via sbatch --export (single-valued — no commas, so it
# survives --export's comma delimiter):
#   sbatch --export=ALL,MODEL_PATH=Qwen/Qwen3-8B,SERVED_MODEL_NAME=Qwen3-8B \
#       scripts/rl/slurm_baseline.sh

#SBATCH --job-name=ofg-base
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=72

set -euo pipefail

# Parameters (override with sbatch --export)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-1.7B}"
N_SEEDS="${N_SEEDS:-3}"
# Per-job port: SLURM may pack several 1-GPU jobs onto one 4-GPU node, so a
# fixed port would collide across concurrent baselines. Derive from the job id
# (consecutive ids → distinct ports). Each job still has its own GPU + its own
# host loopback, so 127.0.0.1:<port> is unambiguous.
VLLM_PORT="${VLLM_PORT:-$(( 8000 + (${SLURM_JOB_ID:-0} % 1500) ))}"
SYS_PROMPT="${SYS_PROMPT:-examples/prompts/financial_competitive_injected_diverse.txt}"
# Context budget: vLLM --max-model-len is the TOTAL (prompt + completion). It MUST
# be >= max_input_tokens + max_output_tokens, else any request asking for
# max_output_tokens of completion is rejected with ContextLengthExceeded. Qwen3's
# native window is 40960 (all sizes); we use it fully = 8192 input + 32768 output.
# max_output 32768 (4x the prior 8192) gives Qwen3 room for <think> + the code so
# the 8B stops truncating mid-thought (OutputLengthExceeded).
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-40960}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-8192}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# A near-full 32768-token generation in eager mode can take many minutes, so lift the
# per-trial wall-clock ceiling accordingly (was 900s).
PER_TRIAL_TIMEOUT="${PER_TRIAL_TIMEOUT:-1800}"

# Conda environment
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniforge3}"
CONDA_ENV="${CONDA_ENV:-openfingym}"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/bin/activate" "${CONDA_ENV}"

# Repo root: prefer the sbatch submit dir (usage assumes `sbatch` from the
# repo root) since under sbatch BASH_SOURCE points into SLURM's spool dir.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
SKYRL_DIR="${SKYRL_DIR:?set SKYRL_DIR to your SkyRL checkout (needed for its examples/ package)}"
TASKS_ROOT="${TASKS_ROOT:-${PROJECT_DIR}/tasks}"

nvidia-smi --query-gpu=name,memory.total --format=csv || true

mkdir -p /tmp/skyrl_examples_only.${SLURM_JOB_ID:-local}
ln -sfn "${SKYRL_DIR}/examples" /tmp/skyrl_examples_only.${SLURM_JOB_ID:-local}/examples
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/src:/tmp/skyrl_examples_only.${SLURM_JOB_ID:-local}:${PYTHONPATH:-}"

# Empty the auto-pipe __init__.py shims so curated trial code imports
# without dragging in langchain/ccxt (not installed in the eval env).
for f in src/openfinai_pipeline/__init__.py \
         src/openfinai_pipeline/benchmark/__init__.py \
         src/openfinai_pipeline/corpus/__init__.py; do
    : > "${PROJECT_DIR}/$f"
done

# HF cache. Compute nodes are often network-isolated: pre-download the model
# into HF_HOME (e.g. on a login node) and run offline. Set HF_HUB_OFFLINE=0
# TRANSFORMERS_OFFLINE=0 if the nodes can reach the Hub.
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/data/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# The system TMPDIR may resolve to a per-node scratch that is NOT created on
# every compute node — standalone `vllm serve` then crashes binding its ZMQ
# IPC socket there (and torch-inductor caching there too). The RL path dodges
# this because Ray redirects to /tmp; we must do it explicitly.
# Pin TMPDIR + the compile caches to a writable /tmp dir (also --enforce-eager).
export TMPDIR="/tmp/ofg_tmp_${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/inductor"
export TRITON_CACHE_DIR="${TMPDIR}/triton"
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

# Per-job verifier registry isolation (the repo is on a shared filesystem, so
# concurrent baseline jobs scoring the SAME task would otherwise clobber
# each other's verifier.url with a 127.0.0.1:<port> local to another node).
# run_trial honors this var when spawning the host verifier.
export OPENFINAI_VERIFIER_RUN_OUTPUT_ROOT="${PROJECT_DIR}/data/run_output/verifier_runs/${SLURM_JOB_ID:-local}"
mkdir -p "${OPENFINAI_VERIFIER_RUN_OUTPUT_ROOT}"

# run_trial reads $HARBOR_TRIAL_CONFIG and uses ONLY its `environment` block
# (the Singularity env: SIF + conda bind + PATH). agent/provider/model stay
# CLI-driven below.
export HARBOR_TRIAL_CONFIG="${HARBOR_TRIAL_CONFIG:-${PROJECT_DIR}/config/harbor_trial_rl_singularity.yaml}"

cd "${PROJECT_DIR}"
echo "=== baseline ${SERVED_MODEL_NAME} (${MODEL_PATH}) | job ${SLURM_JOB_ID:-(none)} on $(hostname) ==="

OUT="${PROJECT_DIR}/data/run_output/baselines/${SERVED_MODEL_NAME}-${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}"

# Serve the OpenAI-compatible endpoint
echo "[serve] vllm serve ${MODEL_PATH} as ${SERVED_MODEL_NAME} on :${VLLM_PORT}"
vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --enforce-eager \
    > "${OUT}/vllm_serve.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT

echo "[serve] waiting for /v1/models ..."
ready=0
for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        ready=1; echo "[serve] ready after ~$((i*5))s"; break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "[serve] vLLM process exited early — tail of vllm_serve.log:"
        tail -60 "${OUT}/vllm_serve.log"; exit 1
    fi
    sleep 5
done
if [[ "${ready}" != "1" ]]; then
    echo "[serve] vLLM not ready after 900s — tail of vllm_serve.log:"
    tail -60 "${OUT}/vllm_serve.log"; exit 1
fi
curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" | head -c 400; echo

# Run the no-RL baseline evaluation
# hosted_vllm requires nested model_info (token limits + costs); pass it as
# a JSON --ak value (run_trial._parse_kv json-decodes {...}).
MODEL_INFO="{\"max_input_tokens\":${MAX_INPUT_TOKENS},\"max_output_tokens\":${MAX_OUTPUT_TOKENS},\"input_cost_per_token\":0,\"output_cost_per_token\":0}"
echo "[eval] baseline.py: 5 test tasks, n_seeds=${N_SEEDS}, prompt=${SYS_PROMPT}"
python -m openfinai_skyrl.eval.baseline \
    --mode single \
    --provider hosted_vllm \
    --model "${SERVED_MODEL_NAME}" \
    --tasks-root "${TASKS_ROOT}" \
    --test-tasks \
        1_commodity_logreturn_forecasting \
        1_crypto_logreturn_forecasting \
        1_equity_logreturn_forecasting \
        1_treasury_variant1_forecasting \
        1_fx_variant1_logreturn_forecasting \
    --system-prompt-path "${SYS_PROMPT}" \
    --output-dir "${OUT}" \
    --n-seeds "${N_SEEDS}" \
    --timeout-sec "${PER_TRIAL_TIMEOUT}" \
    --ak "api_base=http://127.0.0.1:${VLLM_PORT}/v1" \
    --ak "model_info=${MODEL_INFO}"

echo "[done] ${SERVED_MODEL_NAME} baseline -> ${OUT}/summary.json"
