"""Adapt SkyRL's HarborGenerator for OpenFinGym trials.

Adds per-task data mounts, verifier credentials, and loss-to-reward mapping.
SkyRL's examples package must be importable.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from examples.train_integrations.harbor.harbor_generator import (
    HarborGenerator,
)


# Limit verifier-environment logging to once per task.
_VERIFIER_ENV_LOGGED: set[str] = set()


class _TaskAwareTrialConfigDict(dict):
    """Inject task data and verifier settings when ``task`` is assigned.

    SkyRL deep-copies the template before assignment; deepcopy preserves this
    subclass, making each injection isolated and idempotent.
    """

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        if key != "task" or not isinstance(value, dict):
            return
        task_path = value.get("path")
        if not task_path:
            return
        self._inject_data_mount(task_path)
        self._inject_verifier_env(task_path)

    def _environment_section(self) -> dict | None:
        """Return the ``environment`` mapping, creating an empty one if absent.

        Returns ``None`` only if ``environment`` exists but is not a mapping
        (a malformed template) — callers then skip injection rather than
        clobbering it.
        """
        env = self.get("environment")
        if env is None:
            env = {}
            super().__setitem__("environment", env)
            return env
        if not isinstance(env, dict):
            return None
        return env

    def _inject_data_mount(self, task_path: Any) -> None:
        try:
            data_dir = (Path(task_path) / "environment" / "data").resolve()
        except (TypeError, OSError):
            return
        if not data_dir.is_dir():
            return

        env = self._environment_section()
        if env is None:
            return

        mounts = env.get("mounts")
        if mounts is None:
            mounts = []
            env["mounts"] = mounts
        elif not isinstance(mounts, list):
            return

        if any(
            isinstance(m, dict) and m.get("target") == "/data" for m in mounts
        ):
            return

        mounts.append(
            {
                "type": "bind",
                "source": data_dir.as_posix(),
                "target": "/data",
            }
        )

    def _inject_verifier_env(self, task_path: Any) -> None:
        """Inject a live prewarmed verifier endpoint into the environment.

        Stale or absent registry entries are ignored. URLs are translated for
        the container network, and per-job registry roots isolate concurrent
        runs on different nodes.
        """
        try:
            task_dir = Path(task_path).resolve()
        except (TypeError, OSError):
            return

        try:
            from openfinai_harbor.verifier import registry
            from openfinai_harbor.verifier.client import container_url_for
        except Exception:  # openfinai_harbor not importable — nothing to do
            return

        task_id = task_dir.name
        run_output_root = os.environ.get("OPENFINAI_VERIFIER_RUN_OUTPUT_ROOT")
        state_dir = registry.state_dir_for_task(
            task_id=task_id,
            run_output_root=Path(run_output_root) if run_output_root else None,
        )
        info = registry.read_url_file(state_dir)
        url = info.get("url") if info else None
        token = info.get("token") if info else None
        pid = info.get("pid") if info else None
        if not (isinstance(url, str) and isinstance(token, str)):
            self._log_verifier_once(
                task_id,
                "no prewarmed verifier registry found — submissions will "
                "fail and reward will be 0; run prewarm_verifiers.sh first",
                level="warning",
            )
            return
        if isinstance(pid, int) and not registry.pid_is_alive(pid):
            self._log_verifier_once(
                task_id,
                f"verifier registry is stale (pid {pid} dead) — skipping "
                "env injection; reward will be 0",
                level="warning",
            )
            return

        env = self._environment_section()
        if env is None:
            return
        existing = env.get("env")
        if isinstance(existing, dict) and existing.get("VERIFIER_URL"):
            return  # static override wins

        if not isinstance(existing, dict):
            existing = {}
            env["env"] = existing
        existing["VERIFIER_URL"] = container_url_for(url)
        existing["VERIFIER_TOKEN"] = token
        # Unique per trial: no host-side /register is required (the verifier
        # auto-records a heartbeat on /submit), and a fresh id avoids the
        # (agent_id, submission_id) idempotency cache colliding across trials.
        existing.setdefault("AGENT_ID", f"rl-{uuid.uuid4().hex}")
        self._log_verifier_once(
            task_id, f"wired host verifier {existing['VERIFIER_URL']}"
        )

    @staticmethod
    def _log_verifier_once(task_id: str, message: str, *, level: str = "debug") -> None:
        if task_id in _VERIFIER_ENV_LOGGED:
            return
        _VERIFIER_ENV_LOGGED.add(task_id)
        getattr(logger, level, logger.debug)(
            f"[OpenFinAIGenerator] task={task_id}: {message}"
        )


class OpenFinAIGenerator(HarborGenerator):
    """HarborGenerator + per-task /data mount injection + RL reward shaping."""

    # GRPO maximizes rewards, while the verifier returns a loss. Negate valid
    # losses, place the zero/failure sentinel below them, and cap outliers.
    # This transform stays project-local because Harbor also serves tasks whose
    # verifier rewards are already maximization targets.
    _LOSS_CAP = 5.0

    @classmethod
    def _to_rl_reward(cls, verifier_reward: Any) -> float:
        try:
            loss = float(verifier_reward)
        except (TypeError, ValueError):
            return -(cls._LOSS_CAP + 1.0)
        if loss <= 0.0:
            return -(cls._LOSS_CAP + 1.0)
        return -min(loss, cls._LOSS_CAP)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Wrap the template dict in our subclass so deepcopy preserves the
        # intercepting __setitem__.  Upstream uses ``deepcopy(self._harbor_trial_config_template)``
        # in _harbor_agent_loop; the deep-copied object inherits this class.
        if not isinstance(self._harbor_trial_config_template, _TaskAwareTrialConfigDict):
            self._harbor_trial_config_template = _TaskAwareTrialConfigDict(
                self._harbor_trial_config_template
            )

    async def _harbor_agent_loop(self, *args: Any, **kwargs: Any):
        """Run the upstream trial loop, then remap the lower-is-better verifier
        loss into a maximize-friendly RL reward (see ``_to_rl_reward``)."""
        traj = await super()._harbor_agent_loop(*args, **kwargs)
        if traj is not None:
            traj.reward = self._to_rl_reward(traj.reward)
        return traj
