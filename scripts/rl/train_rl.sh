#!/bin/bash
# Online RL training with SkyRL, Harbor, and SingleShotLLMAgent.
#
# Requires NVIDIA GPUs and a supported Harbor sandbox. Singularity/Apptainer
# is the HPC default; use harbor_trial_config.environment.type=docker locally.
# Put the SkyRL checkout root on PYTHONPATH because HarborGenerator is under
# its examples tree:
#   PYTHONPATH=../SkyRL-main:$PYTHONPATH bash scripts/rl/train_rl.sh
#
# SkyRL colocates vLLM with policy workers. SingleShotLLMAgent emits one-step
# trajectories that Harbor executes and scores.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-0.6B}"
VLLM_PORT="${VLLM_PORT:-8000}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
N_SAMPLES="${N_SAMPLES:-4}"            # trajectories per prompt for advantage estimation
ADVANTAGE="${ADVANTAGE:-grpo}"          # grpo, rloo, reinforce_pp

# Colocation requires NUM_INFERENCE_ENGINES * TP_SIZE <= NUM_POLICY_GPUS.
NUM_POLICY_GPUS="${NUM_POLICY_GPUS:-8}"
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-4}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"

TRAIN_DATA="${TRAIN_DATA:-[${PROJECT_DIR}/tasks]}"
PREWARM_ROOT="${PREWARM_ROOT:-${PROJECT_DIR}/tasks}"

# KL loss requires memory for a frozen reference model.
USE_KL_LOSS="${USE_KL_LOSS:-false}"

# CPU offload frees GPU memory for vLLM but increases host RAM use. If disabled,
# lower GPU_MEM_UTIL so the optimizer and vLLM can coexist.
OFFLOAD_OPT_AFTER_STEP="${OFFLOAD_OPT_AFTER_STEP:-true}"

# Keep CKPT_PATH on shared storage, not quota-limited $HOME. The default resume
# mode continues the newest checkpoint after resubmission.
RUN_NAME="${RUN_NAME:-openfinai_rl_run}"
PROJECT_NAME="${PROJECT_NAME:-openfinai_rl}"
CKPT_PATH="${CKPT_PATH:-${PROJECT_DIR}/data/run_output/rl_runs/${RUN_NAME}}"
EXPORT_PATH="${EXPORT_PATH:-${CKPT_PATH}/exports}"
CKPT_INTERVAL="${CKPT_INTERVAL:-50}"      # resumable train ckpt every N steps
HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:-250}"  # HF-format snapshot for eval every N
MAX_CKPTS="${MAX_CKPTS:-3}"               # keep last N resumable ckpts (-1=all)
EPOCHS="${EPOCHS:-1}"                      # passes over the task set (drives #steps)
RESUME_MODE="${RESUME_MODE:-latest}"      # auto-resume from newest ckpt in CKPT_PATH

echo "=== OpenFinGym Online RL Training ==="
echo "Model:                ${MODEL_PATH}"
echo "Advantage:            ${ADVANTAGE}"
echo "Samples/prompt:       ${N_SAMPLES}"
echo "Policy GPUs:          ${NUM_POLICY_GPUS}"
echo "Inference engines:    ${NUM_INFERENCE_ENGINES} (TP=${TP_SIZE})"
echo ""

# Idempotently prewarm the task verifiers.
bash "${SCRIPT_DIR}/prewarm_verifiers.sh" "${PREWARM_ROOT}"
echo ""

python -m openfinai_skyrl.entrypoints.main_rl \
    data.train_data="${TRAIN_DATA}" \
    trainer.policy.model.path="${MODEL_PATH}" \
    trainer.strategy=fsdp \
    trainer.placement.colocate_all=true \
    trainer.placement.policy_num_nodes=1 \
    trainer.placement.policy_num_gpus_per_node="${NUM_POLICY_GPUS}" \
    trainer.placement.ref_num_nodes=1 \
    trainer.placement.ref_num_gpus_per_node="${NUM_POLICY_GPUS}" \
    trainer.algorithm.max_seq_len="${MAX_SEQ_LEN}" \
    trainer.algorithm.advantage_estimator="${ADVANTAGE}" \
    trainer.algorithm.loss_reduction=token_mean \
    trainer.algorithm.use_kl_loss="${USE_KL_LOSS}" \
    trainer.policy.optimizer_config.offload_after_step="${OFFLOAD_OPT_AFTER_STEP}" \
    trainer.eval_before_train=false \
    trainer.eval_interval=-1 \
    trainer.train_batch_size="${TRAIN_BATCH_SIZE}" \
    trainer.policy_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.run_name="${RUN_NAME}" \
    trainer.ckpt_path="${CKPT_PATH}" \
    trainer.export_path="${EXPORT_PATH}" \
    trainer.ckpt_interval="${CKPT_INTERVAL}" \
    trainer.hf_save_interval="${HF_SAVE_INTERVAL}" \
    trainer.max_ckpts_to_keep="${MAX_CKPTS}" \
    trainer.epochs="${EPOCHS}" \
    trainer.resume_mode="${RESUME_MODE}" \
    generator.inference_engine.served_model_name="${SERVED_MODEL_NAME}" \
    generator.inference_engine.http_endpoint_port="${VLLM_PORT}" \
    generator.inference_engine.num_engines="${NUM_INFERENCE_ENGINES}" \
    generator.inference_engine.tensor_parallel_size="${TP_SIZE}" \
    generator.inference_engine.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.async_engine=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.enable_http_endpoint=true \
    generator.batched=false \
    generator.n_samples_per_prompt="${N_SAMPLES}" \
    generator.apply_overlong_filtering=true \
    generator.step_wise_trajectories=true \
    generator.merge_stepwise_output=true
