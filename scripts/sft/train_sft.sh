#!/bin/bash
# SkyRL SFT trainer wrapper; defers to config/sft.yaml, env vars override.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/openfinai-no-mps-${USER:-user}-$$}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

export CUDA_MPS_PIPE_DIRECTORY
export SKYRL_DEBUG_FSDP_INIT="${SKYRL_DEBUG_FSDP_INIT:-0}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

EXTRA_ARGS=("$@")
MODE="${MODE:-single}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/config/sft.yaml}"

DATASET_DIR="${DATASET_DIR:-}"
DATASET_SPLIT="${DATASET_SPLIT:-}"
MESSAGES_KEY="${MESSAGES_KEY:-}"
MODEL_PATH="${MODEL_PATH:-}"
STRATEGY="${STRATEGY:-}"
NUM_STEPS="${NUM_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
MICRO_BATCH="${MICRO_BATCH:-}"
MAX_LENGTH="${MAX_LENGTH:-}"
CKPT_DIR="${CKPT_DIR:-}"
CPU_OFFLOAD="${CPU_OFFLOAD:-}"
NUM_NODES="${NUM_NODES:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-}"
LR="${LR:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
WARMUP_STEPS="${WARMUP_STEPS:-}"
MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:-}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-}"
GRADIENT_CHECKPOINTING_USE_REENTRANT="${GRADIENT_CHECKPOINTING_USE_REENTRANT:-}"
FLASH_ATTN="${FLASH_ATTN:-}"
USE_SAMPLE_PACKING="${USE_SAMPLE_PACKING:-}"
PROJECT_NAME="${PROJECT_NAME:-}"
RUN_NAME="${RUN_NAME:-}"
LOGGER_NAME="${LOGGER_NAME:-}"
CKPT_INTERVAL="${CKPT_INTERVAL:-}"

USE_LORA="${USE_LORA:-}"
LORA_RANK="${LORA_RANK:-}"
LORA_ALPHA="${LORA_ALPHA:-}"
LORA_TARGET="${LORA_TARGET:-}"

show_value() {
    local value="$1"
    local fallback="$2"
    if [[ -n "${value}" ]]; then
        printf '%s\n' "${value}"
    else
        printf '%s\n' "${fallback}"
    fi
}

append_override() {
    local key="$1"
    local value="$2"
    if [[ -n "${value}" ]]; then
        CLI_ARGS+=("${key}=${value}")
    fi
}

# parse_cli requires --config as the first positional arg, before any KEY=VALUE.
CLI_ARGS=(-m openfinai_skyrl.entrypoints.main_sft --mode "${MODE}")
if [[ -f "${CONFIG_PATH}" ]]; then
    CLI_ARGS+=(--config "${CONFIG_PATH}")
fi

append_override "dataset_name" "${DATASET_DIR}"
append_override "dataset_split" "${DATASET_SPLIT}"
append_override "messages_key" "${MESSAGES_KEY}"
append_override "strategy" "${STRATEGY}"
append_override "model.path" "${MODEL_PATH}"
append_override "num_steps" "${NUM_STEPS}"
append_override "batch_size" "${BATCH_SIZE}"
append_override "micro_train_batch_size_per_gpu" "${MICRO_BATCH}"
append_override "max_length" "${MAX_LENGTH}"
append_override "optimizer_config.lr" "${LR}"
append_override "optimizer_config.weight_decay" "${WEIGHT_DECAY}"
append_override "optimizer_config.num_warmup_steps" "${WARMUP_STEPS}"
append_override "placement.num_nodes" "${NUM_NODES}"
append_override "placement.num_gpus_per_node" "${GPUS_PER_NODE}"
append_override "fsdp_config.cpu_offload" "${CPU_OFFLOAD}"
append_override "gradient_checkpointing" "${GRADIENT_CHECKPOINTING}"
append_override "gradient_checkpointing_use_reentrant" "${GRADIENT_CHECKPOINTING_USE_REENTRANT}"
append_override "flash_attn" "${FLASH_ATTN}"
append_override "use_sample_packing" "${USE_SAMPLE_PACKING}"
append_override "ckpt_path" "${CKPT_DIR}"
append_override "ckpt_interval" "${CKPT_INTERVAL}"
append_override "max_ckpts_to_keep" "${MAX_CKPTS_TO_KEEP}"
append_override "logger" "${LOGGER_NAME}"
append_override "project_name" "${PROJECT_NAME}"
append_override "run_name" "${RUN_NAME}"

if [[ "${USE_LORA}" == "false" ]]; then
    CLI_ARGS+=("model.lora.rank=0")
else
    append_override "model.lora.rank" "${LORA_RANK}"
    append_override "model.lora.alpha" "${LORA_ALPHA}"
    append_override "model.lora.target_modules" "${LORA_TARGET}"
fi

echo "=== Experiment ==="
echo "Type:       SkyRL SFT train (mode=${MODE})"
echo "Config:     $(show_value "${CONFIG_PATH}" "<none>")"
echo "Run:        $(show_value "${RUN_NAME}" "<from config>")"
echo "Model:      $(show_value "${MODEL_PATH}" "<from config>")"
echo "Dataset:    $(show_value "${DATASET_DIR}" "<from config>") [split=$(show_value "${DATASET_SPLIT}" "<from config>")]"
echo "Strategy:   $(show_value "${STRATEGY}" "<from config>")"
echo "Steps:      $(show_value "${NUM_STEPS}" "<from config>")"
echo "Batch:      global=$(show_value "${BATCH_SIZE}" "<from config>") micro=$(show_value "${MICRO_BATCH}" "<from config>")"
echo "MaxLength:  $(show_value "${MAX_LENGTH}" "<from config>")"
echo "LR:         $(show_value "${LR}" "<from config>") (warmup_steps=$(show_value "${WARMUP_STEPS}" "<from config>"))"
echo "Placement:  nodes=$(show_value "${NUM_NODES}" "<from config>") gpus_per_node=$(show_value "${GPUS_PER_NODE}" "<from config>") cuda_visible=${CUDA_VISIBLE_DEVICES:-all}"
echo "LoRA:       $(show_value "${USE_LORA}" "<from config>") (rank=$(show_value "${LORA_RANK}" "<from config>") alpha=$(show_value "${LORA_ALPHA}" "<from config>") target=$(show_value "${LORA_TARGET}" "<from config>"))"
echo "FlashAttn:  $(show_value "${FLASH_ATTN}" "<from config>")"
echo "SamplePack: $(show_value "${USE_SAMPLE_PACKING}" "<from config>")"
echo "GradCkpt:   $(show_value "${GRADIENT_CHECKPOINTING}" "<from config>") (reentrant=$(show_value "${GRADIENT_CHECKPOINTING_USE_REENTRANT}" "<from config>"))"
echo "CPUOffload: $(show_value "${CPU_OFFLOAD}" "<from config>")"
echo "Logger:     $(show_value "${LOGGER_NAME}" "<from config>")"
echo "Project:    $(show_value "${PROJECT_NAME}" "<from config>")"
echo "Output:     $(show_value "${CKPT_DIR}" "<from config>")"
echo "Python:     ${PYTHON_BIN}"
echo ""
echo "=== Progress ==="

"${PYTHON_BIN}" "${CLI_ARGS[@]}" "${EXTRA_ARGS[@]}"
