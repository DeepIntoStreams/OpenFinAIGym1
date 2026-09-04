"""Single-process (non-Ray, non-FSDP) policy dispatch for local single-GPU SFT."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.optim as optim
from loguru import logger
from transformers.trainer import get_scheduler

# ray_setup must precede skyrl.* imports (see openfinai_skyrl.train.config).
from openfinai_skyrl.train import ray_setup as _ray_setup  # noqa: F401

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.ppo_utils import PolicyLossRegistry
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.backends.skyrl_train.workers.worker_utils import BatchIterator, reduce_metrics
from skyrl.train.config import SkyRLTrainConfig

from openfinai_skyrl.train.tokenization import resolve_local_hf_path


@dataclass
class _LocalPolicyComponents:
    model: HFModelWrapper
    optimizer: optim.Optimizer
    scheduler: Any
    device: torch.device


class LocalTrainingStrategy:
    def __init__(self, *, max_norm: float) -> None:
        self.max_norm = max_norm
        self.world_size = 1

    def backward(self, loss: torch.Tensor, model, optimizer: optim.Optimizer, **kwargs) -> None:
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model,
        scheduler,
        name: str = "model",
        **kwargs,
    ) -> Optional[torch.Tensor]:
        if isinstance(model, HFModelWrapper):
            model = model.model

        grad_norm = None
        if self.max_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_norm)

        if grad_norm is not None and not torch.isfinite(grad_norm):
            logger.warning("grad_norm is not finite: {}", grad_norm)
            optimizer.zero_grad()
            return grad_norm

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        return grad_norm

    def save_checkpoint(self, model, ckpt_dir, node_local_rank, optimizer, scheduler, tokenizer):
        output_dir = Path(ckpt_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        training_state = {
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(training_state, output_dir / "trainer_state.pt")
        self.save_hf_model(model, str(output_dir / "hf_model"), tokenizer=tokenizer)

    def load_checkpoint(
        self, model, ckpt_dir, optimizer, scheduler, load_module_strict, load_optimizer_states, load_lr_scheduler_states
    ):
        raise NotImplementedError("Local SFT checkpoint restore is not implemented yet")

    def save_hf_model(self, model, output_dir: str, tokenizer=None, **kwargs):
        if isinstance(model, HFModelWrapper):
            model = model.model

        # Patch PEFT's base_model_name_or_path back to the canonical repo id so
        # the adapter is portable; PEFT otherwise serialises the local cache path.
        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = None  # type: ignore[assignment]
        if PeftModel is not None and isinstance(model, PeftModel):
            try:
                canonical = getattr(model.base_model.model.config, "_name_or_path", None)
                if canonical:
                    for adapter_cfg in model.peft_config.values():
                        adapter_cfg.base_model_name_or_path = str(canonical)
            except Exception:
                pass

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir, safe_serialization=True, **kwargs)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

    def all_reduce(self, data, op: str = "mean"):
        return data


class LocalPolicyDispatch:
    def __init__(self, cfg: SkyRLTrainConfig, *, num_training_steps: int) -> None:
        self.cfg = cfg
        self.policy_loss_fn = PolicyLossRegistry.get(cfg.trainer.algorithm.policy_loss_type)
        self.strategy = LocalTrainingStrategy(max_norm=cfg.trainer.policy.optimizer_config.max_grad_norm)
        self._micro_batches_accumulated = 0
        self._components = self._build_policy_components(num_training_steps=num_training_steps)
        self.model = self._components.model
        self.optimizer = self._components.optimizer
        self.scheduler = self._components.scheduler
        self.device = self._components.device

    def _build_policy_components(self, *, num_training_steps: int) -> _LocalPolicyComponents:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = resolve_local_hf_path(self.cfg.trainer.policy.model.path)
        # bf16 only on Ampere+ (sm_80+). Turing (sm_75, RTX 8000) has no bf16 tensor cores.
        bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device.index or 0)[0] >= 8

        logger.info(
            "Initializing local SkyRL SFT backend on device={} with model={} (resolved={})",
            device,
            self.cfg.trainer.policy.model.path,
            model_path,
        )

        wrapped_model = HFModelWrapper(
            model_path,
            use_flash_attention_2=self.cfg.trainer.flash_attn,
            bf16=bf16,
            lora_rank=self.cfg.trainer.policy.model.lora.rank,
            lora_alpha=self.cfg.trainer.policy.model.lora.alpha,
            lora_dropout=self.cfg.trainer.policy.model.lora.dropout,
            lora_init_method=self.cfg.trainer.policy.model.lora.init_method,
            target_modules=self.cfg.trainer.policy.model.lora.target_modules,
            exclude_modules=self.cfg.trainer.policy.model.lora.exclude_modules,
            sequence_parallel_size=1,
            use_sample_packing=self.cfg.trainer.use_sample_packing,
            use_torch_compile=self.cfg.trainer.policy.use_torch_compile,
            rope_scaling={},
            model_config_kwargs=self.cfg.trainer.policy.model_config_kwargs,
        )

        if self.cfg.trainer.gradient_checkpointing:
            wrapped_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant,
                }
            )

        wrapped_model.to(device)
        optimizer = optim.AdamW(
            wrapped_model.parameters(),
            lr=self.cfg.trainer.policy.optimizer_config.lr,
            betas=tuple(self.cfg.trainer.policy.optimizer_config.adam_betas),
            weight_decay=self.cfg.trainer.policy.optimizer_config.weight_decay,
        )
        scheduler = get_scheduler(
            self.cfg.trainer.policy.optimizer_config.scheduler,
            optimizer,
            num_warmup_steps=self.cfg.trainer.policy.optimizer_config.num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return _LocalPolicyComponents(
            model=wrapped_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    def mark_all_offloaded(self) -> None:
        return None

    def _require_policy_role(self, model: str) -> None:
        if model != "policy":
            raise ValueError(f"Local SFT backend only supports model='policy', got {model!r}")

    def _autocast_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        cap = torch.cuda.get_device_capability(self.device.index or 0)[0]
        dtype = torch.bfloat16 if cap >= 8 else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _forward_backward_micro(
        self,
        experience,
        *,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[dict[str, Any]] = None,
    ) -> dict[str, float]:
        experience.to_device(self.device)
        self.model.train()

        current_loss_fn = self.policy_loss_fn if loss_fn is None else PolicyLossRegistry.get(loss_fn)
        resolved_loss_name = loss_fn or self.cfg.trainer.algorithm.policy_loss_type

        if resolved_loss_name != "cross_entropy":
            raise ValueError(f"Local SFT backend only supports cross_entropy, got {resolved_loss_name!r}")

        sequences = experience.sequences
        attention_mask = experience.attention_mask
        loss_mask = experience.loss_mask
        action_mask = experience.action_mask
        num_actions = experience.num_actions

        with self._autocast_context():
            action_log_probs, output = self.model(
                sequences,
                num_actions,
                attention_mask=attention_mask,
                temperature=self.cfg.trainer.algorithm.temperature,
                return_output=True,
                compute_entropy=True,
                entropy_requires_grad=self.cfg.trainer.algorithm.use_entropy_loss,
            )
            policy_loss, loss_metrics = current_loss_fn(
                action_log_probs,
                experience.action_log_probs,
                experience.advantages,
                config=self.cfg.trainer.algorithm,
                loss_mask=loss_mask,
                rollout_logprobs=experience.rollout_logprobs,
            )

        self.strategy.backward(policy_loss, self.model, self.optimizer)

        with torch.no_grad():
            elementwise_loss = -action_log_probs
            if loss_mask is not None:
                elementwise_loss = elementwise_loss * loss_mask

        loss_fn_outputs = []
        for row_idx in range(action_log_probs.shape[0]):
            if action_mask is not None:
                valid_len = int(action_mask[row_idx].sum().item())
            elif loss_mask is not None:
                valid_len = int(loss_mask[row_idx].sum().item())
            else:
                valid_len = action_log_probs.shape[1]

            loss_fn_outputs.append(
                {
                    "logprobs": action_log_probs[row_idx, :valid_len].detach().cpu().tolist(),
                    "elementwise_loss": elementwise_loss[row_idx, :valid_len].detach().cpu().tolist(),
                }
            )

        status = {
            "loss": policy_loss.item(),
            "response_length": num_actions,
            "lr": self.scheduler.get_last_lr()[0],
            "loss_fn_outputs": loss_fn_outputs,
        }
        for key, value in loss_metrics.items():
            status[f"loss_metrics/{key}"] = value

        if self.cfg.trainer.algorithm.use_entropy_loss:
            entropy = masked_mean(output["entropy"][:, -num_actions - 1 : -1], loss_mask)
            status["policy_entropy"] = entropy.item()

        return status

    def forward_backward(
        self,
        model: str,
        data: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[dict[str, Any]] = None,
    ) -> dict[str, float]:
        self._require_policy_role(model)

        all_metrics: dict[str, list[float]] = {}
        all_loss_fn_outputs = []
        for micro_batch in BatchIterator(
            data,
            self.cfg.trainer.micro_train_batch_size_per_gpu,
            drop_last=False,
        ):
            metrics = self._forward_backward_micro(
                micro_batch,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            )
            self._micro_batches_accumulated += 1
            all_loss_fn_outputs.extend(metrics.pop("loss_fn_outputs", []))
            for key, value in metrics.items():
                all_metrics.setdefault(key, []).append(value)

        result = reduce_metrics(all_metrics)
        if all_loss_fn_outputs:
            result["loss_fn_outputs"] = all_loss_fn_outputs
        return result

    def optim_step(self, model: str) -> Optional[float]:
        self._require_policy_role(model)

        if self._micro_batches_accumulated > 0:
            # Known caveat: SkyRL's cross_entropy_loss returns a SUM over
            # loss_mask=1 tokens, so this 1/M scaling is a "mean of per-microbatch
            # sums", not a true per-token mean. Smaller last micro-batches are
            # slightly over-weighted; switching to per-token normalisation would
            # need a matching LR retune so we keep this as the default.
            scale = 1.0 / self._micro_batches_accumulated
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)

        grad_norm = self.strategy.optimizer_step(
            self.optimizer,
            self.model,
            self.scheduler,
            name="policy",
        )
        self._micro_batches_accumulated = 0
        if isinstance(grad_norm, torch.Tensor):
            return float(grad_norm.item())
        return float(grad_norm) if grad_norm is not None else None

    def save_checkpoint(self, model: str, ckpt_dir: str, tokenizer=None) -> None:
        self._require_policy_role(model)
        self.strategy.save_checkpoint(
            self.model,
            ckpt_dir=ckpt_dir,
            node_local_rank=0,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            tokenizer=tokenizer,
        )

    def save_hf_model(self, model: str, export_dir: str, tokenizer) -> None:
        self._require_policy_role(model)
        self.strategy.save_hf_model(self.model, export_dir, tokenizer=tokenizer)
