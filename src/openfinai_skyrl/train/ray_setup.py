"""Ray initialisation for SFT training.

Includes a compatibility shim for SkyRL 0.1.0: SkyRL imports
``PlacementGroupSchedulingStrategy`` from the old
``ray.util.placement_group`` path while newer Ray exposes it from
``ray.util.scheduling_strategies``. We patch the old path so SkyRL's
``workers.worker`` module imports cleanly.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import ray
from loguru import logger
from ray._private import services as _ray_services
from ray.util.scheduling_strategies import (
    PlacementGroupSchedulingStrategy as _PlacementGroupSchedulingStrategy,
)

_ray_pg_module = importlib.import_module("ray.util.placement_group")
if not hasattr(_ray_pg_module, "PlacementGroupSchedulingStrategy"):
    _ray_pg_module.PlacementGroupSchedulingStrategy = _PlacementGroupSchedulingStrategy

from skyrl.backends.skyrl_train.utils.ppo_utils import sync_registries
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import prepare_runtime_environment


def initialize_local_ray(cfg: SkyRLTrainConfig) -> None:
    """Start a single-node Ray cluster wired to the project's PYTHONPATH."""
    env_vars = prepare_runtime_environment(cfg)
    project_root = Path(__file__).resolve().parents[3]
    src_root = project_root / "src"
    pythonpath_entries = [str(project_root), str(src_root)]
    existing_pythonpath = env_vars.get("PYTHONPATH") or os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.extend(p for p in existing_pythonpath.split(os.pathsep) if p)
    env_vars["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_entries))
    if os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        env_vars["CUDA_MPS_PIPE_DIRECTORY"] = os.environ["CUDA_MPS_PIPE_DIRECTORY"]

    ray_address = os.environ.get("OPENFINAI_SFT_RAY_ADDRESS", "local")
    node_ip = os.environ.get("OPENFINAI_SFT_RAY_NODE_IP", "127.0.0.1")

    if ray_address == "local" and node_ip in {"127.0.0.1", "localhost", "::1"}:
        original_resolve_ip = _ray_services.resolve_ip_for_localhost

        def _preserve_loopback(host: str) -> str:
            if host in {"127.0.0.1", "localhost", "::1"}:
                return "127.0.0.1"
            return original_resolve_ip(host)

        _ray_services.resolve_ip_for_localhost = _preserve_loopback

    init_kwargs: dict[str, Any] = {
        "address": ray_address,
        "runtime_env": {"env_vars": env_vars},
        "log_to_driver": True,
        "ignore_reinit_error": True,
    }
    if ray_address == "local":
        init_kwargs["_node_ip_address"] = node_ip

    logger.info(
        "Initializing Ray with address={} node_ip={} pythonpath={}",
        ray_address,
        init_kwargs.get("_node_ip_address", "auto"),
        env_vars["PYTHONPATH"],
    )
    ray.init(**init_kwargs)
    sync_registries()


__all__ = ["initialize_local_ray"]
