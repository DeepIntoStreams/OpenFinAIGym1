"""Online RL (PPO/GRPO) entrypoint for OpenFinGym + Harbor + SkyRL."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import ray
import yaml

from skyrl.train.config import GeneratorConfig, SkyRLTrainConfig, get_config_as_yaml_str
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.rate_limiter import RateLimiterConfig
from skyrl.train.utils.utils import initialize_ray

from openfinai_skyrl.openfinai_generator import OpenFinAIGenerator
from openfinai_skyrl.openfinai_dataset import OpenFinAITaskDataset


_REPO_ROOT = Path(__file__).parent.parent.parent.parent

# HARBOR_TRIAL_CONFIG env override lets clusters where apptainer is blocked
# (e.g. a PodSecurity `baseline` namespace) point at a podman yaml without code changes.
_HARBOR_CONFIG_FROM_ENV = os.environ.get("HARBOR_TRIAL_CONFIG", "").strip()
if _HARBOR_CONFIG_FROM_ENV:
    _candidate = Path(_HARBOR_CONFIG_FROM_ENV)
    HARBOR_DEFAULT_CONFIG = _candidate if _candidate.is_absolute() else (_REPO_ROOT / _candidate)
else:
    HARBOR_DEFAULT_CONFIG = _REPO_ROOT / "config" / "harbor_trial_rl.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _cap_ray_object_store() -> None:
    """Bound Ray's plasma object store via OPENFINAI_RAY_OBJECT_STORE_GB.

    SkyRL's ``initialize_ray`` calls ``ray.init()`` with no
    ``object_store_memory``, so Ray auto-sizes the store to ~30% of the
    node's PHYSICAL RAM, clamped to its ~200GB ceiling. On a shared SLURM
    node that ~200GB reservation is charged to the job's (much smaller)
    memory cgroup and OOM-kills the step at startup — independent of model
    size (an 8B policy is only ~16GB, and on the GPU). Our RL rollouts are
    KB-scale, so a few GB suffices. Patch ``ray.init`` (whose attribute
    SkyRL resolves at call time) to inject the cap. Opt-in: unset leaves
    upstream behavior untouched for non-HPC callers.
    """
    raw = os.environ.get("OPENFINAI_RAY_OBJECT_STORE_GB", "").strip()
    if not raw:
        return
    try:
        cap_bytes = int(float(raw) * 1e9)
    except ValueError:
        return
    if cap_bytes <= 0:
        return
    _orig_init = ray.init

    def _capped_init(*args: Any, **kwargs: Any):
        kwargs.setdefault("object_store_memory", cap_bytes)
        return _orig_init(*args, **kwargs)

    ray.init = _capped_init


@dataclass
class OpenFinAIGeneratorConfig(GeneratorConfig):
    rate_limit: RateLimiterConfig = field(default_factory=RateLimiterConfig)


@dataclass
class OpenFinAISkyRLConfig(SkyRLTrainConfig):
    harbor_trial_config: Dict[str, Any] = field(default_factory=dict)
    generator: OpenFinAIGeneratorConfig = field(default_factory=OpenFinAIGeneratorConfig)


class OpenFinAIExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return OpenFinAIGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        ds = OpenFinAITaskDataset(task_dirs=self.cfg.data.train_data)
        assert len(ds) >= self.cfg.trainer.train_batch_size, (
            f"Dataset ({len(ds)} tasks) must be >= train_batch_size ({self.cfg.trainer.train_batch_size})"
        )
        return ds

    def get_eval_dataset(self):
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            return OpenFinAITaskDataset(task_dirs=self.cfg.data.val_data)
        return None


def _sync_vllm_server_port(cfg) -> None:
    """Bind the vLLM server to the SAME port the agent's api_base targets.

    SkyRL derives the agent's api_base from
    ``generator.inference_engine.http_endpoint_port`` (HarborGenerator), but the
    server's listen port is computed from the module constant
    ``inference_servers.setup.VLLM_START_PORT`` independently — they coincide
    only because both default to the same value. To run concurrent jobs on one
    node we set a per-job ``http_endpoint_port``; without this sync the server
    keeps binding ``VLLM_START_PORT`` while the agent dials the per-job port, so
    rollouts get no connection and degenerate to ~1 token. Patch the constant to
    match. Runs inside the Ray worker, where ``create_inference_servers`` reads
    it; the VLLMServerActor receives the resulting ``start_port`` as an arg.
    """
    try:
        port = int(cfg.generator.inference_engine.http_endpoint_port)
    except (TypeError, ValueError, AttributeError):
        return
    try:
        from skyrl.backends.skyrl_train.inference_servers import setup as _vllm_setup
        _vllm_setup.VLLM_START_PORT = port
    except Exception:
        return


def _gate_checkpoint_cadence(cfg) -> None:
    """Make ``trainer.ckpt_interval`` the TRUE save cadence (every N steps).

    SkyRL's train loop saves on ``is_epoch_end OR step % ckpt_interval == 0``
    (trainer.py), and ``is_epoch_end`` fires every ``len(dataloader)`` steps
    (~6 here: 27 tasks / batch 4). So the epoch-end trigger forces a save every
    ~6 steps regardless of ckpt_interval — there's no config to widen it. We
    gate ``RayPPOTrainer.save_checkpoints`` so it only runs on multiples of
    ckpt_interval (plus the final step), making the interval the real cadence
    and keeping unlimited retention (max_ckpts_to_keep=-1) manageable. Runs in
    the Ray worker where the trainer is instantiated. No-op when interval<=0.
    """
    try:
        every = int(cfg.trainer.ckpt_interval)
    except (TypeError, ValueError, AttributeError):
        return
    if every <= 0:
        return
    from skyrl.train.trainer import RayPPOTrainer

    _orig_save = RayPPOTrainer.save_checkpoints

    def _gated_save(self, *args: Any, **kwargs: Any):
        total = getattr(self, "total_training_steps", None)
        is_final = total is not None and self.global_step >= total
        if self.global_step % every == 0 or is_final:
            return _orig_save(self, *args, **kwargs)
        return None

    RayPPOTrainer.save_checkpoints = _gated_save


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    _sync_vllm_server_port(cfg)
    _gate_checkpoint_cadence(cfg)
    exp = OpenFinAIExp(cfg)
    exp.run()


def main() -> None:
    cfg = OpenFinAISkyRLConfig.from_cli_overrides(sys.argv[1:])

    if HARBOR_DEFAULT_CONFIG.exists():
        with open(HARBOR_DEFAULT_CONFIG) as f:
            defaults = yaml.safe_load(f)
        cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be explicitly set for Harbor training"
        )
    _cap_ray_object_store()
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
