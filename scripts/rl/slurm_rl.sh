#!/bin/bash
# Online RL (full-param GRPO, SkyRL+Harbor+vLLM) on a SLURM cluster with
# Singularity/Apptainer, for ONE SFT-merged init model. SIZE-keyed GPU layout;
# per-(SIZE,BATCH) MODEL/RUN/CKPT.
# Reward: openfinai_generator.OpenFinAIGenerator remaps the verifier's
# lower-is-better total_loss -> maximize-friendly RL reward (see that file).
# resume_mode=latest -> a re-submitted job auto-continues from CKPT_PATH's newest
# checkpoint (survives the 24h cap). ckpt every 50 steps, ~1000 steps (150 epochs
# over the 27-task set at batch 4), console logger -> per-step reward+timing in .out.
#
# Site-specific knobs (env vars; see slurm_smoke.sh): SKYRL_DIR (required),
# CONDA_ROOT, CONDA_ENV, PROJECT_DIR, HF_HOME, CUDA_HOME. MODEL_PATH defaults
# to the SFT-merged model staged under data/run_output/rl_init/ (see below).
#
#   sbatch --job-name=ofg-rl-b2-8b --gpus=4 --export=ALL,SIZE=8b,BATCH=b2 scripts/rl/slurm_rl.sh

#SBATCH --job-name=ofg-rl
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-gpu=72

set -euo pipefail

SIZE="${SIZE:?set SIZE=1.7b|4b|8b}"
BATCH="${BATCH:?set BATCH=b1|b2}"
TAG="${SIZE/./p}"

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniforge3}"
CONDA_ENV="${CONDA_ENV:-openfingym}"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/bin/activate" "${CONDA_ENV}"

# Repo root: prefer the sbatch submit dir (usage assumes `sbatch` from the
# repo root) since under sbatch BASH_SOURCE points into SLURM's spool dir.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
SKYRL_DIR="${SKYRL_DIR:?set SKYRL_DIR to your SkyRL checkout (needed for its examples/ package)}"
cd "${PROJECT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# SkyRL examples shim (OpenFinAIGenerator imports examples.train_integrations.harbor)
# WITHOUT putting the SkyRL checkout root on PYTHONPATH (would shadow the installed skyrl).
mkdir -p "/tmp/skyrl_examples_only.${SLURM_JOB_ID:-local}"
ln -sfn "${SKYRL_DIR}/examples" "/tmp/skyrl_examples_only.${SLURM_JOB_ID:-local}/examples"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/src:/tmp/skyrl_examples_only.${SLURM_JOB_ID:-local}:${PYTHONPATH:-}"

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
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
# Ray sizes its plasma store at ~30% of PHYSICAL node RAM (clamped ~200GB),
# charged to our cgroup -> OOM-kills the step at startup on shared nodes.
# main_rl.py reads this and caps it; rollouts are KB-scale so 32GB is ample.
export OPENFINAI_RAY_OBJECT_STORE_GB="${OPENFINAI_RAY_OBJECT_STORE_GB:-32}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TMPDIR="/tmp/ofg_tmp_${SLURM_JOB_ID:-$$}"
mkdir -p "${TMPDIR}"
: "${WANDB_API_KEY:=disabled}"; export WANDB_API_KEY
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HARBOR_TRIAL_CONFIG="${HARBOR_TRIAL_CONFIG:-${PROJECT_DIR}/config/harbor_trial_rl_singularity.yaml}"
# Per-job vLLM HTTP endpoint port. The engine binds it AND HarborGenerator
# derives the agent's api_base from it (harbor_generator.py:205,230), so a
# unique port lets multiple RL jobs share a node without colliding on the
# default 8000 (which kills the later job's vLLM router). SLURM_JOB_IDs are
# sequential, so the modulo keeps each concurrent job distinct.
export VLLM_PORT="${VLLM_PORT:-$(( 10000 + ${SLURM_JOB_ID:-0} % 50000 ))}"
# Per-job verifier registry isolation (concurrent RL jobs score the same tasks).
export OPENFINAI_VERIFIER_RUN_OUTPUT_ROOT="${PROJECT_DIR}/data/run_output/verifier_runs/rl_${SLURM_JOB_ID:-local}"
mkdir -p "${OPENFINAI_VERIFIER_RUN_OUTPUT_ROOT}"

# Per-model settings
# NOTE: staged merged dirs are named qwen3_<tag>_sftb1_merged / _sftb2_merged
# (the hf_merged output of the SFT eval, copied under data/run_output/rl_init/).
export MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/data/run_output/rl_init/qwen3_${TAG}_sft${BATCH}_merged}"
export SERVED_MODEL_NAME="qwen3-${TAG}-${BATCH}-rl"
export RUN_NAME="${RUN_NAME:-rl_qwen3_${TAG}_${BATCH}}"
export CKPT_PATH="${CKPT_PATH:-${PROJECT_DIR}/data/run_output/rl_runs/${RUN_NAME}}"
export TRAIN_DATA="${TRAIN_DATA:-[${PROJECT_DIR}/tasks/rl_train_set]}"
export PREWARM_ROOT="${PREWARM_ROOT:-${PROJECT_DIR}/tasks/rl_train_set}"
export N_SAMPLES="${N_SAMPLES:-8}"            # GRPO group size (variance for advantage)
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export EPOCHS="${EPOCHS:-150}"                # ~1000 steps over 27 tasks @ batch 4
export CKPT_INTERVAL="${CKPT_INTERVAL:-20}"   # resumable train ckpt every 20 steps
export HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:-50}"   # eval-able HF snapshot every 50
export MAX_CKPTS="${MAX_CKPTS:-3}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
export ADVANTAGE="${ADVANTAGE:-grpo}"
export USE_KL_LOSS="${USE_KL_LOSS:-false}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"

# GPU layout (full-param FSDP2 GRPO + colocated vLLM; engines*TP <= policy_gpus).
case "${SIZE}" in
    1.7b) export NUM_POLICY_GPUS="${NUM_POLICY_GPUS:-2}" NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-2}" TP_SIZE="${TP_SIZE:-1}" ;;
    4b)   export NUM_POLICY_GPUS="${NUM_POLICY_GPUS:-4}" NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-2}" TP_SIZE="${TP_SIZE:-2}" ;;
    8b)   export NUM_POLICY_GPUS="${NUM_POLICY_GPUS:-4}" NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-2}" TP_SIZE="${TP_SIZE:-2}" ;;
    *) echo "unknown SIZE: ${SIZE}" >&2; exit 1 ;;
esac

echo "=== git HEAD ==="; git log -1 --oneline 2>&1 || true
echo "=== RL ${RUN_NAME} | init=${MODEL_PATH} | job ${SLURM_JOB_ID:-(none)} on $(hostname) ==="
echo "    gpus(policy=${NUM_POLICY_GPUS} engines=${NUM_INFERENCE_ENGINES} TP=${TP_SIZE}) n_samples=${N_SAMPLES} epochs=${EPOCHS} ckpt_every=${CKPT_INTERVAL}"
echo "    ckpt_path=${CKPT_PATH}  train_data=${TRAIN_DATA}"

exec bash scripts/rl/train_rl.sh
