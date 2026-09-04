"""FastAPI app factory for curated-task verifier services.

Family handlers mount their own routes. Shared lifecycle machinery handles
auth and audits; state, scoring, and audit locks protect mutable resources.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI

from openfinai_harbor.verifier import registry
from openfinai_harbor.verifier._lifecycle import (
    AgentRecord,
    bearer_dependency,
    mount_lifecycle_routes,
    start_lifecycle_thread,
)
from openfinai_harbor.verifier.auth import RateLimiter
from openfinai_harbor.verifier.curated import handlers
from openfinai_harbor.verifier.curated.models import CuratedInfoResponse
from openfinai_harbor.verifier.curated.task_loader import (
    CuratedTaskSpec,
    import_task_class,
)


logger = logging.getLogger("openfinai_harbor.verifier.curated")


@dataclass
class CuratedVerifierConfig:
    """Per-task configuration captured at startup.

    Built by the curated branch in ``__main__.py`` from CLI flags +
    ``task.toml``. Identical lifecycle/auth knobs to
    :class:`openfinai_harbor.verifier.server.VerifierConfig`; the
    family-specific bits live in ``spec``.
    """

    spec: CuratedTaskSpec
    state_dir: Path                         # data/run_output/verifier/<task_id>/
    run_dir: Path                           # state_dir/runs/<utc_ts>/
    token: str
    despawn_grace_min: float
    heartbeat_timeout_sec: float = 90.0
    liveness_poll_sec: float = 30.0
    max_requests_per_minute: int = 60
    submission_cache_size: int = 64

    @property
    def task_id(self) -> str:
        return self.spec.task_id

    @property
    def family(self) -> str:
        return self.spec.family

    @property
    def task_dir(self) -> Path:
        return self.spec.task_dir


@dataclass
class CuratedVerifierState:
    """Per-process mutable state. Guarded by ``state_lock``.

    Family handlers stash arbitrary per-process state in
    ``handler_state`` (keyed by handler module name) — e.g. forecasting
    caches the held-out gt + dataset, trading caches a prototype task
    instance, realtime forecasting caches the open ledger handle.
    """

    config: CuratedVerifierConfig
    task_cls: type                          # imported curated task class
    handler_module: Any                     # the family handler (forecasting / trading / ...)
    reward_names: List[str] = field(default_factory=list)
    agents: Dict[str, AgentRecord] = field(default_factory=dict)
    submission_cache: "OrderedDict[str, Any]" = field(default_factory=OrderedDict)
    rate_limiter: RateLimiter = field(init=False)
    last_idle_at: Optional[float] = None
    started_at: float = field(default_factory=time.time)
    started_at_iso: str = field(default_factory=registry.utc_now_iso)
    n_total_submissions: int = 0
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    score_lock: threading.Lock = field(default_factory=threading.Lock)
    audit_lock: threading.Lock = field(default_factory=threading.Lock)
    lifecycle_stop: threading.Event = field(default_factory=threading.Event)
    handler_state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rate_limiter = RateLimiter(
            max_requests=self.config.max_requests_per_minute,
            window_sec=60.0,
        )


def create_curated_app(config: CuratedVerifierConfig) -> FastAPI:
    """Build a FastAPI app for one curated task.

    Imports the task class, picks the family handler, mounts the shared
    lifecycle endpoints, then asks the handler to register its routes.
    Any failure (bad task.toml, broken curated task.py, missing required
    data cache) raises here — uvicorn never starts listening, and the
    spawn coordinator surfaces a clean error to the caller.
    """
    handler_module = handlers.handler_for(config.family)
    if handler_module is None:
        raise RuntimeError(
            f"no curated handler registered for family={config.family!r}; "
            f"available: {sorted(handlers.FAMILY_HANDLERS)}"
        )
    task_cls = import_task_class(config.spec)

    state = CuratedVerifierState(
        config=config,
        task_cls=task_cls,
        handler_module=handler_module,
        reward_names=list(getattr(handler_module, "REWARD_NAMES", []) or []),
    )

    # Family-specific startup hook. Failures fail the spawn — better to crash
    # now than at first submission.
    if hasattr(handler_module, "on_startup"):
        try:
            handler_module.on_startup(state)
        except Exception:
            logger.exception(
                "curated handler on_startup failed for family=%s", config.family
            )
            raise

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        thread = start_lifecycle_thread(
            state, app, logger=logger, log_label="curated verifier"
        )
        logger.info(
            "curated verifier ready: task_id=%s family=%s reward_names=%s",
            config.task_id,
            config.family,
            state.reward_names,
        )
        try:
            yield
        finally:
            state.lifecycle_stop.set()
            thread.join(timeout=2.0)
            registry.remove_url_file(config.state_dir)
            logger.info(
                "curated verifier shut down: task_id=%s", config.task_id
            )

    app = FastAPI(
        title=f"OpenFinGym Curated Verifier ({config.task_id})",
        description=(
            f"Curated-task verifier for family={config.family}. "
            "Owns the curated task class; submissions are dispatched "
            "through it for scoring."
        ),
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.config = config
    app.state.state = state
    bearer = bearer_dependency(state)

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        with state.state_lock:
            return {
                "ok": True,
                "task_id": config.task_id,
                "family": config.family,
                "n_active_agents": len(state.agents),
                "n_total_submissions": state.n_total_submissions,
            }

    @app.get("/info", response_model=CuratedInfoResponse)
    async def info(_: str = Depends(bearer)) -> CuratedInfoResponse:
        with state.state_lock:
            n_active = len(state.agents)
        return CuratedInfoResponse(
            task_id=config.task_id,
            family=config.family,  # type: ignore[arg-type]  — Literal narrowed at config-build time
            reward_names=list(state.reward_names),
            n_active_agents=n_active,
            started_at_utc=state.started_at_iso,
            despawn_grace_min=config.despawn_grace_min,
        )

    mount_lifecycle_routes(app, state, bearer=bearer, logger=logger)

    handler_module.register_routes(app, state, bearer=bearer)

    return app


__all__ = [
    "CuratedVerifierConfig",
    "CuratedVerifierState",
    "create_curated_app",
    "logger",
]
