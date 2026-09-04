"""SkyRL training-config construction from the operator's raw config dict."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

# Order matters: ``ray_setup`` patches ``ray.util.placement_group`` so that
# SkyRL 0.1.0's ``train.utils`` import path resolves. Must precede the
# ``skyrl.train.utils`` import below.
from openfinai_skyrl.train import ray_setup as _ray_setup  # noqa: F401

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import validate_cfg

from openfinai_skyrl.train.tokenization import get_nested


@dataclass
class SFTRuntimeConfig:
    """Operator-facing knobs not represented in SkyRL's own config tree."""

    dataset_name: str
    dataset_split: str
    messages_key: str
    num_steps: int
    batch_size: int
    max_length: int


def uses_local_backend(strategy: str) -> bool:
    """``True`` when SFT runs in-process without Ray/FSDP."""
    return strategy == "local"


def _apply_dataclass_overrides(target: Any, overrides: dict[str, Any]) -> None:
    """Recursively assign ``overrides`` onto a SkyRL config dataclass tree."""
    if not overrides:
        return
    valid_fields = {field.name for field in fields(target)}
    for key, value in overrides.items():
        if key not in valid_fields:
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_dataclass_overrides(current, value)
        else:
            setattr(target, key, value)


def build_train_config(raw: dict[str, Any]) -> tuple[SkyRLTrainConfig, SFTRuntimeConfig]:
    """Translate the OmegaConf-parsed CLI dict into SkyRL + runtime configs."""
    cfg = SkyRLTrainConfig()

    cfg.trainer.strategy = str(raw.get("strategy", cfg.trainer.strategy))
    cfg.trainer.policy.model.path = str(
        get_nested(raw, "model.path", cfg.trainer.policy.model.path)
    )
    cfg.trainer.policy.model.lora.rank = int(
        get_nested(raw, "model.lora.rank", cfg.trainer.policy.model.lora.rank)
    )
    cfg.trainer.policy.model.lora.alpha = int(
        get_nested(raw, "model.lora.alpha", cfg.trainer.policy.model.lora.alpha)
    )
    cfg.trainer.policy.model.lora.target_modules = str(
        get_nested(raw, "model.lora.target_modules", cfg.trainer.policy.model.lora.target_modules)
    )
    cfg.trainer.policy.model.lora.dropout = float(
        get_nested(raw, "model.lora.dropout", cfg.trainer.policy.model.lora.dropout)
    )

    batch_size = int(raw.get("batch_size", 1))
    micro_batch_size = int(raw.get("micro_train_batch_size_per_gpu", 1))
    max_length = int(raw.get("max_length", 2048))
    num_steps = int(raw.get("num_steps", 100))

    cfg.trainer.seed = int(raw.get("seed", cfg.trainer.seed))
    cfg.trainer.train_batch_size = batch_size
    cfg.trainer.policy_mini_batch_size = batch_size
    cfg.trainer.micro_train_batch_size_per_gpu = micro_batch_size
    cfg.trainer.micro_forward_batch_size_per_gpu = micro_batch_size
    cfg.trainer.max_prompt_length = max_length
    cfg.trainer.project_name = str(raw.get("project_name", cfg.trainer.project_name))
    cfg.trainer.run_name = str(raw.get("run_name", cfg.trainer.run_name))
    cfg.trainer.logger = str(raw.get("logger", "console"))
    cfg.trainer.ckpt_path = str(raw.get("ckpt_path", cfg.trainer.ckpt_path))
    cfg.trainer.ckpt_interval = int(raw.get("ckpt_interval", cfg.trainer.ckpt_interval))
    cfg.trainer.max_ckpts_to_keep = int(raw.get("max_ckpts_to_keep", cfg.trainer.max_ckpts_to_keep))
    cfg.trainer.gradient_checkpointing = bool(
        raw.get("gradient_checkpointing", cfg.trainer.gradient_checkpointing)
    )
    cfg.trainer.gradient_checkpointing_use_reentrant = bool(
        raw.get(
            "gradient_checkpointing_use_reentrant",
            cfg.trainer.gradient_checkpointing_use_reentrant,
        )
    )
    cfg.trainer.eval_interval = -1
    cfg.trainer.eval_before_train = False
    cfg.trainer.epochs = 1
    # FA2 + sample packing both default on now that the real flash-attn 2.8.3
    # wheel is installed against torch 2.8 cu12. Override either via the yaml
    # if you want to fall back to eager attention (e.g. on non-Ampere/Hopper).
    cfg.trainer.flash_attn = bool(raw.get("flash_attn", True))
    cfg.trainer.use_sample_packing = bool(raw.get("use_sample_packing", True))
    cfg.trainer.disable_fast_tokenizer = False
    cfg.trainer.resume_mode = None
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.placement.policy_num_nodes = int(
        get_nested(
            raw,
            "placement.num_nodes",
            get_nested(raw, "placement.policy_num_nodes", cfg.trainer.placement.policy_num_nodes),
        )
    )
    cfg.trainer.placement.policy_num_gpus_per_node = int(
        get_nested(
            raw,
            "placement.num_gpus_per_node",
            get_nested(
                raw,
                "placement.policy_num_gpus_per_node",
                cfg.trainer.placement.policy_num_gpus_per_node,
            ),
        )
    )
    cfg.trainer.algorithm.policy_loss_type = "cross_entropy"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.algorithm.temperature = 1.0
    cfg.trainer.algorithm.loss_reduction = "token_mean"
    cfg.trainer.policy.fsdp_config.cpu_offload = bool(
        get_nested(raw, "fsdp_config.cpu_offload", cfg.trainer.policy.fsdp_config.cpu_offload)
    )
    cfg.trainer.policy.optimizer_config.lr = float(
        get_nested(raw, "optimizer_config.lr", cfg.trainer.policy.optimizer_config.lr)
    )
    adam_betas = get_nested(raw, "optimizer_config.adam_betas", cfg.trainer.policy.optimizer_config.adam_betas)
    cfg.trainer.policy.optimizer_config.adam_betas = [float(beta) for beta in adam_betas]
    cfg.trainer.policy.optimizer_config.weight_decay = float(
        get_nested(raw, "optimizer_config.weight_decay", cfg.trainer.policy.optimizer_config.weight_decay)
    )
    cfg.trainer.policy.optimizer_config.max_grad_norm = float(
        get_nested(raw, "optimizer_config.max_grad_norm", cfg.trainer.policy.optimizer_config.max_grad_norm)
    )
    cfg.trainer.policy.optimizer_config.num_warmup_steps = int(
        get_nested(raw, "optimizer_config.num_warmup_steps", cfg.trainer.policy.optimizer_config.num_warmup_steps)
    )
    cfg.trainer.policy.optimizer_config.scheduler = str(
        get_nested(raw, "optimizer_config.scheduler", cfg.trainer.policy.optimizer_config.scheduler)
    )
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.max_input_length = max_length
    cfg.generator.sampling_params.max_generate_length = max_length

    top_level_megatron = raw.get("megatron_config")
    if isinstance(top_level_megatron, dict):
        _apply_dataclass_overrides(cfg.trainer.policy.megatron_config, top_level_megatron)

    policy_megatron = get_nested(raw, "policy.megatron_config", None)
    if isinstance(policy_megatron, dict):
        _apply_dataclass_overrides(cfg.trainer.policy.megatron_config, policy_megatron)

    trainer_policy_megatron = get_nested(raw, "trainer.policy.megatron_config", None)
    if isinstance(trainer_policy_megatron, dict):
        _apply_dataclass_overrides(cfg.trainer.policy.megatron_config, trainer_policy_megatron)

    if uses_local_backend(cfg.trainer.strategy):
        if cfg.trainer.placement.policy_num_nodes != 1:
            raise ValueError("strategy=local only supports placement.num_nodes=1")
        if cfg.trainer.placement.policy_num_gpus_per_node != 1:
            raise ValueError("strategy=local only supports placement.num_gpus_per_node=1")
        if cfg.trainer.micro_train_batch_size_per_gpu < 1:
            raise ValueError("micro_train_batch_size_per_gpu must be >= 1")
    else:
        validate_cfg(cfg)

    total_gpus = (
        cfg.trainer.placement.policy_num_nodes * cfg.trainer.placement.policy_num_gpus_per_node
    )
    if total_gpus > 1 and batch_size % total_gpus != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by total_gpus={total_gpus} "
            f"({cfg.trainer.placement.policy_num_nodes} node(s) × "
            f"{cfg.trainer.placement.policy_num_gpus_per_node} GPU(s) per node). "
            f"MeshDispatch requires the batch to split evenly across data-parallel ranks. "
            f"Set batch_size to a multiple of {total_gpus} (e.g. BATCH_SIZE={total_gpus})."
        )

    runtime = SFTRuntimeConfig(
        dataset_name=str(raw.get("dataset_name", "data/sft_dataset")),
        dataset_split=str(raw.get("dataset_split", "train")),
        messages_key=str(raw.get("messages_key", "messages")),
        num_steps=num_steps,
        batch_size=batch_size,
        max_length=max_length,
    )
    return cfg, runtime


__all__ = ["SFTRuntimeConfig", "build_train_config", "uses_local_backend"]
