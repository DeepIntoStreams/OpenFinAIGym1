#!/bin/bash
# Single-GPU RL smoke run (any 1-GPU node).
#
# Targets: Qwen3-1.7B, offline_crypto_forecasting only.
# Memory layout for a single H100 80GB with colocate_all=true:
#   - Policy (1.7B fp16)             ~3.4 GB
#   - vLLM weights + KV cache        ~25 GB at gpu_memory_utilization=0.5
#   - Optimizer state (Adam, FSDP)   ~13 GB (≈4x param size in fp16)
#   - Activations + buffers          ~10-20 GB
# With use_kl_loss=false (default) there is no reference-model copy.
#
# Usage:
#   bash scripts/rl/smoke_rl_single_gpu.sh
#
# Override anything via env vars before invocation, e.g.
#   TRAIN_BATCH_SIZE=2 N_SAMPLES=2 bash scripts/rl/smoke_rl_single_gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Single-task smoke
export TRAIN_DATA="${TRAIN_DATA:-[${PROJECT_DIR}/tasks/offline_crypto_forecasting]}"
export PREWARM_ROOT="${PREWARM_ROOT:-${PROJECT_DIR}/tasks/offline_crypto_forecasting}"

# Qwen3-1.7B
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-1.7B}"

# 1 GPU, no tensor parallelism, no separate inference engines beyond one
export NUM_POLICY_GPUS="${NUM_POLICY_GPUS:-1}"
export NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-1}"
export TP_SIZE="${TP_SIZE:-1}"
# Lower vLLM memory share so the policy + optimizer fit on the same GPU.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"

# Smaller batch (single GPU can't sustain the default 4 with a 1.7B model;
# also can't exceed task count — we only have one task in this smoke run).
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export N_SAMPLES="${N_SAMPLES:-2}"

# KL loss requires a frozen ref-model copy on the same GPU — skip for the
# smoke run, follow upstream run_codecontest.sh which also runs without it.
export USE_KL_LOSS="${USE_KL_LOSS:-false}"

echo "=== Single-GPU smoke configuration ==="
echo "Model:        ${MODEL_PATH}"
echo "Task:         ${TRAIN_DATA}"
echo "GPUs:         ${NUM_POLICY_GPUS} (TP=${TP_SIZE}, engines=${NUM_INFERENCE_ENGINES})"
echo "vLLM mem:     ${GPU_MEM_UTIL}"
echo "Batch:        ${TRAIN_BATCH_SIZE} (n_samples=${N_SAMPLES})"
echo "use_kl_loss:  ${USE_KL_LOSS}"
echo ""

exec bash "${SCRIPT_DIR}/train_rl.sh"
