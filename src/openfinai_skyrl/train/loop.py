"""SFT training loop, dispatch factory, and checkpoint I/O."""
from __future__ import annotations

import importlib
import random
from pathlib import Path
from typing import Any, Iterator, Protocol

import ray
from loguru import logger
from transformers import AutoTokenizer

# ray_setup must precede skyrl.* imports (see openfinai_skyrl.train.config).
from openfinai_skyrl.train import ray_setup as _ray_setup  # noqa: F401

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.train.config import SkyRLTrainConfig

from openfinai_skyrl.train.config import SFTRuntimeConfig, uses_local_backend
from openfinai_skyrl.train.dataset import SFTExample, collate_sft_batch
from openfinai_skyrl.train.policy import LocalPolicyDispatch
from openfinai_skyrl.train.tokenization import hf_load_kwargs, resolve_local_hf_path


class SFTDispatch(Protocol):
    """Common policy-dispatch surface used by the training loop."""

    def forward_backward(
        self,
        model: str,
        data: TrainingInputBatch,
        loss_fn: str | None = None,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> dict[str, float]: ...

    def optim_step(self, model: str) -> float | None: ...

    def save_checkpoint(self, model: str, ckpt_dir: str, tokenizer=None) -> None: ...

    def save_hf_model(self, model: str, export_dir: str, tokenizer) -> None: ...

    def mark_all_offloaded(self) -> None: ...


def load_tokenizer(model_path: str) -> AutoTokenizer:
    """HF AutoTokenizer with pad_token defaulted to eos_token (SkyRL convention)."""
    model_source = resolve_local_hf_path(model_path)
    logger.info("Loading tokenizer for {} via {}", model_path, model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source, **hf_load_kwargs(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_policy_worker(strategy: str):
    if strategy in {"fsdp", "fsdp2"}:
        module = importlib.import_module("skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker")
        return module.PolicyWorker
    if strategy == "megatron":
        try:
            module = importlib.import_module("skyrl.backends.skyrl_train.workers.megatron.megatron_worker")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "strategy=megatron requires SkyRL Megatron dependencies to be installed "
                "(for example megatron-core / megatron-bridge)."
            ) from exc
        return module.PolicyWorker
    raise ValueError(
        f"Unsupported SFT strategy={strategy}. Use one of local, fsdp, fsdp2, or megatron."
    )


def build_policy_dispatch(cfg: SkyRLTrainConfig, num_steps: int) -> SFTDispatch:
    """Instantiate the right dispatcher for the configured strategy."""
    if uses_local_backend(cfg.trainer.strategy):
        return LocalPolicyDispatch(cfg, num_training_steps=num_steps)

    num_nodes = int(cfg.trainer.placement.policy_num_nodes)
    num_gpus_per_node = int(cfg.trainer.placement.policy_num_gpus_per_node)
    if num_nodes < 1:
        raise ValueError("placement.num_nodes must be >= 1 for native SkyRL SFT backends")
    if num_gpus_per_node < 1:
        raise ValueError("placement.num_gpus_per_node must be >= 1 for native SkyRL SFT backends")

    policy_worker = _resolve_policy_worker(cfg.trainer.strategy)
    actor_group = PPORayActorGroup(
        cfg.trainer,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        ray_actor_type=policy_worker,
        num_gpus_per_actor=1.0,
        colocate_all=False,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        record_memory=cfg.trainer.policy.record_memory,
    )
    ray.get(actor_group.async_init_model(cfg.trainer.policy.model.path, num_training_steps=num_steps))
    dispatch = WorkerDispatch(cfg, policy_actor_group=actor_group)
    dispatch.mark_all_offloaded()
    return dispatch


def save_checkpoint(dispatch: SFTDispatch, cfg: SkyRLTrainConfig, tokenizer: AutoTokenizer, step: int) -> None:
    """Snapshot the policy at ``<ckpt_path>/checkpoints/step_<N>/``."""
    ckpt_root = Path(cfg.trainer.ckpt_path) / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    dispatch.save_checkpoint("policy", str(ckpt_root / f"step_{step:06d}"), tokenizer=tokenizer)


def save_final_hf_export(dispatch: SFTDispatch, cfg: SkyRLTrainConfig, tokenizer: AutoTokenizer) -> None:
    """Write the final HF (or PEFT-adapter) export under ``<ckpt_path>/hf_final/``."""
    export_dir = Path(cfg.trainer.ckpt_path) / "hf_final"
    export_dir.mkdir(parents=True, exist_ok=True)
    dispatch.save_hf_model("policy", str(export_dir), tokenizer)


def _shuffled_epoch_iter(data: list[SFTExample], seed: int) -> Iterator[SFTExample]:
    """Yield ``data`` indefinitely with a fresh permutation at every epoch boundary.

    Uses a deterministic seed so re-running with the same ``trainer.seed``
    reproduces identical batch order; the per-epoch reshuffle prevents the
    model from memorising a fixed step→example mapping (the failure mode of
    ``itertools.cycle`` on a small SFT corpus).
    """
    rng = random.Random(seed)
    while True:
        buf = list(data)
        rng.shuffle(buf)
        yield from buf


def run_sft_training_loop(
    cfg: SkyRLTrainConfig,
    runtime: SFTRuntimeConfig,
    tokenizer: AutoTokenizer,
    examples: list[SFTExample],
    dispatch: SFTDispatch,
) -> list[dict[str, Any]]:
    """Run the SFT loop; returns ``{step, loss, grad_norm, response_length, lr}`` per step."""
    logger.info(
        "Starting SFT loop: strategy={} steps={} batch_size={} micro_batch_size={} max_length={}",
        cfg.trainer.strategy,
        runtime.num_steps,
        runtime.batch_size,
        cfg.trainer.micro_train_batch_size_per_gpu,
        runtime.max_length,
    )

    step_rows: list[dict[str, Any]] = []
    example_iter = _shuffled_epoch_iter(examples, cfg.trainer.seed)
    for step in range(1, runtime.num_steps + 1):
        batch_examples = [next(example_iter) for _ in range(runtime.batch_size)]
        batch = collate_sft_batch(batch_examples, tokenizer)
        status = dispatch.forward_backward("policy", batch, loss_fn="cross_entropy")
        grad_norm = dispatch.optim_step("policy")

        loss_value = status.get("loss", float("nan"))
        lr_value = status.get("lr")
        response_length = batch.metadata["response_length"]
        logger.info(
            "step={} loss={:.4f} grad_norm={} response_length={} lr={}",
            step,
            float(loss_value),
            grad_norm,
            response_length,
            lr_value,
        )
        step_rows.append(
            {
                "step": int(step),
                "loss": float(loss_value),
                "grad_norm": float(grad_norm) if grad_norm is not None else None,
                "response_length": int(response_length),
                "lr": float(lr_value) if lr_value is not None else None,
            }
        )

        if cfg.trainer.ckpt_interval > 0 and step % cfg.trainer.ckpt_interval == 0:
            save_checkpoint(dispatch, cfg, tokenizer, step)

    save_final_hf_export(dispatch, cfg, tokenizer)
    logger.info("SFT training complete; final HF export saved under {}", cfg.trainer.ckpt_path)
    return step_rows


__all__ = [
    "SFTDispatch",
    "build_policy_dispatch",
    "load_tokenizer",
    "run_sft_training_loop",
    "save_checkpoint",
    "save_final_hf_export",
]
