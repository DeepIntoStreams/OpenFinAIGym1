"""Shared lifecycle + auth machinery for the verifier RPC services.

Both the auto-pipe verifier (one ``/submit`` endpoint, evaluator-driven)
and the curated verifier (multiple ``/submit/*`` endpoints, family-
handler-driven) share the same agent-tracking lifecycle, audit log
shape, bearer auth, and despawn-on-idle behavior. This module factors
out those identical pieces so both server.py files become thin
orchestrators around their genuinely different startup / scoring flows.

Duck typing
-----------
Functions here accept ``state: Any`` rather than a specific type
because :class:`~openfinai_harbor.verifier.server.VerifierState`
(auto-pipe) and
:class:`~openfinai_harbor.verifier.curated.server.CuratedVerifierState`
(curated) carry different family-specific fields but the same
lifecycle fields. Required attributes are documented per function.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from openfinai_harbor.verifier import registry
from openfinai_harbor.verifier.auth import constant_time_token_eq
from openfinai_harbor.verifier.models import (
    HeartbeatRequest,
    HeartbeatResponse,
    RegisterRequest,
    RegisterResponse,
)


_BEARER = HTTPBearer(auto_error=False)


# AgentRecord — the per-agent tracking row used by both verifier flavors


@dataclass
class AgentRecord:
    """Per-agent tracking row.

    Held in ``state.agents: Dict[str, AgentRecord]`` and updated under
    ``state.state_lock`` by register/heartbeat/deregister/submit handlers.
    """

    agent_id: str
    registered_at: float
    last_heartbeat: float
    last_submit: Optional[float] = None
    n_submissions: int = 0
    notes: Optional[str] = None


# Bearer auth dependency factory


def bearer_dependency(state: Any):
    """Build a FastAPI dependency that validates ``Authorization: Bearer``.

    Constant-time comparison against ``state.config.token``. Used by
    every authenticated route in both verifier flavors. Returns the raw
    token string so route handlers can pass it on to the rate limiter.
    """

    async def _verify(
        creds: Optional[HTTPAuthorizationCredentials] = Depends(_BEARER),
    ) -> str:
        if creds is None or not creds.credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not constant_time_token_eq(creds.credentials, state.config.token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return creds.credentials

    return _verify


# Audit log + shutdown helpers


def audit_log(state: Any, payload: Dict[str, Any]) -> None:
    """Append a one-line JSON audit record to ``run_dir/submissions.jsonl``.

    Serialises under ``state.audit_lock`` so concurrent submits don't
    interleave bytes within a single record. ``payload`` is JSON-encoded
    with ``default=str`` so non-serialisable values (e.g. datetimes,
    pathlibs) round-trip as their ``__str__``.
    """
    audit_path = registry.submissions_log_path(state.config.run_dir)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, default=str) + "\n"
    with state.audit_lock:
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(line)


def request_shutdown(app: FastAPI) -> None:
    """Signal uvicorn to gracefully exit. Cross-platform.

    ``app.state.server`` is the ``uvicorn.Server`` instance stashed by
    ``__main__.py`` before calling ``server.run()``. Uvicorn polls
    ``server.should_exit`` in its event loop and cleanly tears down. If
    ``server`` is missing (e.g. tests / standalone invocations) we fall
    back to SIGINT-to-self.
    """
    server = getattr(app.state, "server", None)
    if server is not None:
        server.should_exit = True
        return
    try:
        os.kill(os.getpid(), signal.SIGINT)
    except Exception:  # pragma: no cover — best-effort
        pass


def start_lifecycle_thread(
    state: Any,
    app: FastAPI,
    *,
    logger: Optional[logging.Logger] = None,
    log_label: str = "verifier",
) -> threading.Thread:
    """Spawn the daemon that drops dead agents + triggers despawn.

    Polls every ``state.config.liveness_poll_sec``:

    * Drops agents whose ``last_heartbeat`` is older than
      ``state.config.heartbeat_timeout_sec``.
    * Tracks ``state.last_idle_at`` when the active set hits zero.
    * Calls :func:`request_shutdown` once idle exceeds
      ``state.config.despawn_grace_min``.

    ``log_label`` distinguishes auto-pipe vs curated in the despawn log
    line and the thread name (otherwise the lines are identical).
    """
    log = logger or logging.getLogger(__name__)

    def _run() -> None:
        cfg = state.config
        while not state.lifecycle_stop.wait(cfg.liveness_poll_sec):
            now = time.time()
            with state.state_lock:
                stale = [
                    aid
                    for aid, rec in state.agents.items()
                    if (now - rec.last_heartbeat) > cfg.heartbeat_timeout_sec
                ]
                for aid in stale:
                    state.agents.pop(aid, None)
                if not state.agents:
                    if state.last_idle_at is None:
                        state.last_idle_at = now
                else:
                    state.last_idle_at = None
                if state.last_idle_at is not None:
                    idle_for = now - state.last_idle_at
                    if idle_for >= cfg.despawn_grace_min * 60.0:
                        log.info(
                            "%s idle for %.1fs (>= grace %.1fs); shutting down",
                            log_label,
                            idle_for,
                            cfg.despawn_grace_min * 60.0,
                        )
                        state.lifecycle_stop.set()
                        request_shutdown(app)
                        return

    t = threading.Thread(
        target=_run,
        name=f"{log_label.replace(' ', '-')}-lifecycle",
        daemon=True,
    )
    t.start()
    return t


# Endpoint handlers (factored out so tests can drive without TestClient)


def do_register(state: Any, req: RegisterRequest) -> RegisterResponse:
    now = time.time()
    with state.state_lock:
        rec = state.agents.get(req.agent_id)
        if rec is None:
            rec = AgentRecord(
                agent_id=req.agent_id,
                registered_at=now,
                last_heartbeat=now,
                notes=req.notes,
            )
            state.agents[req.agent_id] = rec
        else:
            rec.last_heartbeat = now
            if req.notes is not None:
                rec.notes = req.notes
        state.last_idle_at = None  # presence resets the idle clock
        n_active = len(state.agents)
    return RegisterResponse(
        agent_id=req.agent_id,
        task_id=state.config.task_id,
        registered_at_utc=registry.utc_now_iso(),
        n_active_agents=n_active,
    )


def do_heartbeat(state: Any, req: HeartbeatRequest) -> HeartbeatResponse:
    now = time.time()
    with state.state_lock:
        rec = state.agents.get(req.agent_id)
        if rec is None:
            # Auto-register on heartbeat — keeps clients simple.
            rec = AgentRecord(
                agent_id=req.agent_id,
                registered_at=now,
                last_heartbeat=now,
            )
            state.agents[req.agent_id] = rec
        else:
            rec.last_heartbeat = now
        state.last_idle_at = None
        n_active = len(state.agents)
    return HeartbeatResponse(
        agent_id=req.agent_id,
        last_heartbeat_utc=registry.utc_now_iso(),
        n_active_agents=n_active,
    )


def do_deregister(
    state: Any,
    req: HeartbeatRequest,
    app: FastAPI,
    *,
    logger: Optional[logging.Logger] = None,
) -> HeartbeatResponse:
    """Deregister an agent. If it was the last one and ``despawn_grace_min``
    is zero, schedule an immediate graceful shutdown.

    The shutdown is timer-deferred by 100ms so the response goes out
    before uvicorn stops accepting writes; the lifecycle thread would
    catch this on its next tick anyway, but the immediate path makes
    the ``--despawn-grace-min=0`` contract explicit.
    """
    log = logger or logging.getLogger(__name__)
    now = time.time()
    with state.state_lock:
        state.agents.pop(req.agent_id, None)
        n_active = len(state.agents)
        if n_active == 0:
            state.last_idle_at = now
            immediate = state.config.despawn_grace_min <= 0.0
        else:
            immediate = False
    if immediate:
        log.info(
            "last agent deregistered; despawn_grace_min=0 -> immediate shutdown"
        )
        state.lifecycle_stop.set()
        threading.Timer(0.1, request_shutdown, args=(app,)).start()
    return HeartbeatResponse(
        agent_id=req.agent_id,
        last_heartbeat_utc=registry.utc_now_iso(),
        n_active_agents=n_active,
    )


def record_submit_heartbeat(state: Any, agent_id: str) -> None:
    """Auto-bump heartbeat from inside a submit handler.

    A ``/submit`` IS proof of life — handlers call this so an agent that
    only ever calls submit (no separate heartbeat) doesn't get reaped.
    Also increments ``rec.n_submissions`` and
    ``state.n_total_submissions``.
    """
    now = time.time()
    with state.state_lock:
        rec = state.agents.get(agent_id)
        if rec is None:
            rec = AgentRecord(
                agent_id=agent_id, registered_at=now, last_heartbeat=now
            )
            state.agents[agent_id] = rec
        else:
            rec.last_heartbeat = now
        rec.last_submit = now
        rec.n_submissions += 1
        state.last_idle_at = None
        state.n_total_submissions += 1


# Route mounting


def mount_lifecycle_routes(
    app: FastAPI,
    state: Any,
    *,
    bearer,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Mount ``/register``, ``/heartbeat``, ``/deregister`` on ``app``.

    These three endpoints are byte-identical between the auto-pipe and
    curated verifier flavors. ``/healthz`` and ``/info`` stay in each
    flavor's ``server.py`` because they have flavor-specific response
    shapes (auto-pipe surfaces ``interaction_model`` /
    ``has_held_out_test_gt`` / ``expects_fake_emb``; curated surfaces
    ``family``).
    """

    @app.post("/register", response_model=RegisterResponse)
    async def register(
        req: RegisterRequest, _: str = Depends(bearer)
    ) -> RegisterResponse:
        return do_register(state, req)

    @app.post("/heartbeat", response_model=HeartbeatResponse)
    async def heartbeat(
        req: HeartbeatRequest, _: str = Depends(bearer)
    ) -> HeartbeatResponse:
        return do_heartbeat(state, req)

    @app.post("/deregister", response_model=HeartbeatResponse)
    async def deregister(
        req: HeartbeatRequest, _: str = Depends(bearer)
    ) -> HeartbeatResponse:
        return do_deregister(state, req, app, logger=logger)


__all__ = [
    "AgentRecord",
    "audit_log",
    "bearer_dependency",
    "do_deregister",
    "do_heartbeat",
    "do_register",
    "mount_lifecycle_routes",
    "record_submit_heartbeat",
    "request_shutdown",
    "start_lifecycle_thread",
]
