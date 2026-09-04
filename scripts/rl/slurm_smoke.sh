#!/bin/bash
# Smoke RL job for a SLURM cluster with Singularity/Apptainer: 1 GPU,
# Qwen3-1.7B, tasks/offline_crypto_forecasting only.
#
# Site-specific knobs are env vars (override via sbatch --export=ALL,VAR=val):
#   SKYRL_DIR    (required) SkyRL checkout — only its examples/ subtree is used
#   CONDA_ROOT   conda/miniforge install root        (default ~/miniforge3)
#   CONDA_ENV    env with skyrl/vllm/harbor/ray       (default openfingym)
#   PROJECT_DIR  repo root (default: $SLURM_SUBMIT_DIR, else derived from this file)
#   HF_HOME      HF cache                             (default $PROJECT_DIR/data/hf_cache)
#   CUDA_HOME    CUDA toolkit                         (default /usr/local/cuda)
#
#   sbatch --export=ALL,SKYRL_DIR=$HOME/SkyRL scripts/rl/slurm_smoke.sh

#SBATCH --job-name=ofg-rl-smoke
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=72

set -euo pipefail

# Activate the conda env (skyrl/vllm/harbor/fastapi/uvicorn/ray installed).
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniforge3}"
CONDA_ENV="${CONDA_ENV:-openfingym}"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/bin/activate" "${CONDA_ENV}"

# Repo root. Under sbatch the script runs from SLURM's spool dir, so prefer
# the submit dir (usage above assumes `sbatch` is run from the repo root).
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
SKYRL_DIR="${SKYRL_DIR:?set SKYRL_DIR to your SkyRL checkout (needed for its examples/ package)}"

mkdir -p "${PROJECT_DIR}/data/run_output/rl_trials"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Same PYTHONPATH stratagem as the K8s RL job template (deploy/k8s/): a
# single-link staging dir surfaces SkyRL's `examples/` subtree without
# putting the whole SkyRL src on PYTHONPATH (would shadow the installed skyrl).
mkdir -p /tmp/skyrl_examples_only.${SLURM_JOB_ID:-local}
ln -sfn "${SKYRL_DIR}/examples" \
    /tmp/skyrl_examples_only.${SLURM_JOB_ID:-local}/examples

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/src:/tmp/skyrl_examples_only.${SLURM_JOB_ID:-local}:${PYTHONPATH:-}"

# Same /openfinai-namespace-shim trick as docker/Dockerfile.base: empty
# the auto-pipe __init__.py files so curated trial code can `import
# openfinai_pipeline.benchmark.contracts` without dragging in
# langchain/ccxt/akshare (auto-pipe-only deps not in the training env).
for f in src/openfinai_pipeline/__init__.py \
         src/openfinai_pipeline/benchmark/__init__.py \
         src/openfinai_pipeline/corpus/__init__.py; do
    : > "${PROJECT_DIR}/$f"
done

# Caches: HF + Triton + (NVCC fallback when triton needs to compile)
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/data/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# SkyRL's validate_generator_cfg asserts WANDB_API_KEY is non-empty.
: "${WANDB_API_KEY:=disabled}"
export WANDB_API_KEY WANDB_MODE="${WANDB_MODE:-disabled}"

cd "${PROJECT_DIR}"

echo "=== git HEAD ==="
git log -1 --oneline 2>&1 || true
echo "=== job ${SLURM_JOB_ID:-(no slurm)} on $(hostname) ==="

# Delegate to the existing smoke launcher; it sets MODEL/GPU sizing
# vars then python -m openfinai_skyrl.entrypoints.main_rl.
#
# OVERRIDE the trial config: use the Singularity-specific yaml that swaps
# in the SingularityEnvironment subclass + adds the conda-env bind mount.
# main_rl reads $HARBOR_TRIAL_CONFIG; default is harbor_trial_rl.yaml.
export HARBOR_TRIAL_CONFIG="${HARBOR_TRIAL_CONFIG:-${PROJECT_DIR}/config/harbor_trial_rl_singularity.yaml}"

bash scripts/rl/smoke_rl_single_gpu.sh
