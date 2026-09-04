#!/bin/bash
# Single-GPU LoRA SFT (strategy=local) on a SLURM cluster. Each Qwen3 size
# (1.7B/4B/8B) fits one 80GB-class GPU with LoRA+bf16+grad-ckpt, so every run
# is a 1-GPU job; launch all three concurrently.
#
# strategy=local means in-process training: no Ray, no FSDP, no vLLM, no
# per-trial sandbox — so unlike the RL/baseline jobs this needs neither the
# SkyRL examples shim nor a host verifier nor a Singularity image. The full
# SFT import chain (skyrl.train.* + peft) resolves in the conda env.
#
# Site-specific knobs (env vars; see scripts/rl/slurm_smoke.sh): CONDA_ROOT,
# CONDA_ENV, PROJECT_DIR, HF_HOME. No SKYRL_DIR needed here.
#
# Usage (1.7b first, then 4b/8b):
#   sbatch --job-name=ofg-sft-1p7b --export=ALL,SIZE=1.7b scripts/sft/slurm_sft.sh
#   sbatch --job-name=ofg-sft-4b   --export=ALL,SIZE=4b   scripts/sft/slurm_sft.sh
#   sbatch --job-name=ofg-sft-8b   --export=ALL,SIZE=8b   scripts/sft/slurm_sft.sh
# Any knob is overridable, e.g.  --export=ALL,SIZE=4b,LR=1e-4,NUM_STEPS=92

#SBATCH --job-name=ofg-sft
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=72

set -euo pipefail

# Per-size LoRA and schedule defaults
# Tuned for ~182 single-shot rows; batch=4, packing OFF -> 46 steps/epoch.
SIZE="${SIZE:-1.7b}"
case "${SIZE}" in
    1.7b) DEF_MODEL="Qwen/Qwen3-1.7B"; DEF_RANK=32; DEF_ALPHA=64; DEF_DROPOUT=0.05
          DEF_LR=2e-4;   DEF_STEPS=184; DEF_WARMUP=18; DEF_MICRO=4 ;;   # ~4 epochs
    4b)   DEF_MODEL="Qwen/Qwen3-4B";  DEF_RANK=16; DEF_ALPHA=32; DEF_DROPOUT=0.05
          DEF_LR=1.5e-4; DEF_STEPS=138; DEF_WARMUP=14; DEF_MICRO=2 ;;   # ~3 epochs
    8b)   DEF_MODEL="Qwen/Qwen3-8B";  DEF_RANK=16; DEF_ALPHA=32; DEF_DROPOUT=0.10
          DEF_LR=1e-4;   DEF_STEPS=92;  DEF_WARMUP=9;  DEF_MICRO=1 ;;   # ~2 epochs
    *) echo "unknown SIZE: ${SIZE} (use 1.7b|4b|8b)" >&2; exit 1 ;;
esac

SIZE_TAG="${SIZE/./p}"   # 1.7b -> 1p7b

MODEL_PATH="${MODEL_PATH:-${DEF_MODEL}}"
RUN_NAME="${RUN_NAME:-sft_single_qwen3_${SIZE_TAG}}"
PROJECT_NAME="${PROJECT_NAME:-data/run_output/experiments-sft}"
DATASET_DIR="${DATASET_DIR:-data/run_output/experiments-sft/data/dataset_single_task_split}"
CKPT_DIR="${CKPT_DIR:-data/run_output/experiments-sft/runs/${RUN_NAME}}"

NUM_STEPS="${NUM_STEPS:-${DEF_STEPS}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MICRO_BATCH="${MICRO_BATCH:-${DEF_MICRO}}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
USE_SAMPLE_PACKING="${USE_SAMPLE_PACKING:-false}"   # off -> exact 46 steps/epoch
FLASH_ATTN="${FLASH_ATTN:-true}"                    # flip to false if FA2 misbehaves on your GPU
LR="${LR:-${DEF_LR}}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-${DEF_WARMUP}}"
LORA_RANK="${LORA_RANK:-${DEF_RANK}}"
LORA_ALPHA="${LORA_ALPHA:-${DEF_ALPHA}}"
LORA_DROPOUT="${LORA_DROPOUT:-${DEF_DROPOUT}}"
LORA_TARGET="${LORA_TARGET:-all-linear}"
CKPT_INTERVAL="${CKPT_INTERVAL:-46}"                # ~1 checkpoint / epoch
MAX_CKPTS="${MAX_CKPTS:-8}"                         # keep all epoch ckpts

# Conda environment
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniforge3}"
CONDA_ENV="${CONDA_ENV:-openfingym}"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/bin/activate" "${CONDA_ENV}"

# Repo root: prefer the sbatch submit dir (usage assumes `sbatch` from the
# repo root) since under sbatch BASH_SOURCE points into SLURM's spool dir.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "${PROJECT_DIR}"

# Repo root + curated reward bank importable. NO SkyRL examples shim: SkyRL is
# pip-installed in the conda env; adding a SkyRL checkout root here would
# shadow it.
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/src:${PYTHONPATH:-}"

# Empty the auto-pipe __init__ shims (defensive; same trick as the RL/baseline
# jobs) so any transitive import never drags in langchain/ccxt.
for f in src/openfinai_pipeline/__init__.py \
         src/openfinai_pipeline/benchmark/__init__.py \
         src/openfinai_pipeline/corpus/__init__.py; do
    : > "${PROJECT_DIR}/$f"
done

# HF cache. Compute nodes are often network-isolated: pre-download the base
# models (Qwen3 1.7B/4B/8B) into HF_HOME, e.g. on a login node, and load fully
# offline. Set HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 if the nodes can reach
# the Hub.
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/data/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1                            # console logger -> live .out
# The system TMPDIR may resolve to a per-node scratch that isn't always present;
# pin it (and torch compile caches) to a writable per-job /tmp dir.
export TMPDIR="/tmp/ofg_tmp_${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/inductor"
export TRITON_CACHE_DIR="${TMPDIR}/triton"
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "=== git HEAD ==="; git log -1 --oneline 2>&1 || true
echo "=== SFT ${SIZE} (${MODEL_PATH}) | run=${RUN_NAME} | job ${SLURM_JOB_ID:-(none)} on $(hostname) ==="
echo "schedule: steps=${NUM_STEPS} batch=${BATCH_SIZE} micro=${MICRO_BATCH} max_len=${MAX_LENGTH} pack=${USE_SAMPLE_PACKING} fa2=${FLASH_ATTN}"
echo "optim:    lr=${LR} wd=${WEIGHT_DECAY} warmup=${WARMUP_STEPS}"
echo "lora:     rank=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT} target=${LORA_TARGET}"
echo "ckpt:     ${CKPT_DIR} (every ${CKPT_INTERVAL}, keep ${MAX_CKPTS})"

# Train in-process on one GPU
exec python -m openfinai_skyrl.entrypoints.main_sft \
    --mode single \
    --config config/sft.yaml \
    strategy=local \
    placement.num_nodes=1 \
    placement.num_gpus_per_node=1 \
    model.path="${MODEL_PATH}" \
    model.lora.rank="${LORA_RANK}" \
    model.lora.alpha="${LORA_ALPHA}" \
    model.lora.dropout="${LORA_DROPOUT}" \
    model.lora.target_modules="${LORA_TARGET}" \
    dataset_name="${DATASET_DIR}" \
    dataset_split=train \
    num_steps="${NUM_STEPS}" \
    batch_size="${BATCH_SIZE}" \
    micro_train_batch_size_per_gpu="${MICRO_BATCH}" \
    max_length="${MAX_LENGTH}" \
    use_sample_packing="${USE_SAMPLE_PACKING}" \
    flash_attn="${FLASH_ATTN}" \
    optimizer_config.lr="${LR}" \
    optimizer_config.weight_decay="${WEIGHT_DECAY}" \
    optimizer_config.num_warmup_steps="${WARMUP_STEPS}" \
    ckpt_path="${CKPT_DIR}" \
    ckpt_interval="${CKPT_INTERVAL}" \
    max_ckpts_to_keep="${MAX_CKPTS}" \
    run_name="${RUN_NAME}" \
    project_name="${PROJECT_NAME}" \
    logger=console
