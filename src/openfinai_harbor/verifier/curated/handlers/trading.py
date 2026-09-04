"""Serve held-out-safe, interactive offline trading sessions.

Agents reset a session and advance it one bar per step; future bars never cross
the verifier boundary. Each session gets fresh mutable task state while cached
market data is reused. Terminal scoring remains server-side.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from openfinai_harbor.verifier import registry
from openfinai_harbor.verifier._lifecycle import (
    audit_log,
    record_submit_heartbeat,
)
from openfinai_harbor.verifier.curated._coerce import coerce_scores
from openfinai_harbor.verifier.curated.models import (
    CuratedSessionResetRequest,
    CuratedSessionResetResponse,
    CuratedSessionStepRequest,
    CuratedSessionStepResponse,
)


logger = logging.getLogger(__name__)


# Static fallback covering the four curated TradingTask metrics + inline
# scalar stats; the dry-run probe at startup overrides this dynamically.
REWARD_NAMES = [
    "pnl",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "total_return",
    "mean_return",
    "std_return",
    "num_trades",
]

#: Fallback headline when a task omits an explicit reward.
HEADLINE_METRIC = "pnl"


_HANDLER_KEY = "trading"

#: Interactive-session lifecycle. The verifier holds one live task per open
#: episode between /session/step calls; idle sessions are reaped so a crashed
#: agent can't pin state forever, and a per-agent cap bounds memory.
_SESSION_TTL_SEC = 600.0
_MAX_SESSIONS_PER_AGENT = 8


def on_startup(state) -> None:  # type: CuratedVerifierState
    """Cache the prototype task instance so per-submission spawns are cheap.

    What we cache:

    * ``prototype`` — a single fully-loaded curated trading task. Its
      ``_data`` / ``_ohlcv`` / ``_prices`` get reused by every
      replay-task instance (we copy attribute references after each
      ``__init__``, so we don't pay the CSV load + feature engineer
      cost per submission).
    * ``reward_names`` — populated via a one-step dry-run replay to
      mirror the actual metric set the task's ``evaluate()`` returns.
    """
    from openfinai_harbor.verifier.curated.task_loader import instantiate_task

    cfg = state.config
    spec = cfg.spec

    prototype = instantiate_task(state.task_cls, default_config=spec.default_config)
    prototype.load_data()  # forces _ohlcv / _prices population

    handler_state: Dict[str, Any] = {
        "prototype": prototype,
        # session_id -> live interactive episode (see /session/* endpoints).
        "sessions": {},
    }
    state.handler_state[_HANDLER_KEY] = handler_state

    # Dry-run probe: a single no-op step + evaluate is enough to surface
    # every key from _compute_trading_metrics_from_history.
    try:
        probe = _build_replay_task(state)
        probe.reset()
        first_action = _zero_action_for(probe)
        probe.step(first_action)
        scores = probe.evaluate([first_action])
        if isinstance(scores, dict) and scores:
            # Preserve task-defined insertion order.
            state.reward_names = [
                k for k in scores.keys() if not k.startswith("_")
            ]
    except Exception as exc:
        logger.warning(
            "trading on_startup dry-run probe failed (%s); "
            "falling back to static REWARD_NAMES",
            exc,
        )
        state.reward_names = list(REWARD_NAMES)


def _build_replay_task(state) -> Any:  # type: CuratedVerifierState
    """Instantiate a fresh curated trading task that shares the prototype's
    cached OHLCV. Returns a task with portfolio state at zero.

    We copy ``_data`` / ``_ohlcv`` / ``_prices`` / ``_min_len`` from
    the prototype by reference — these are read-only after load_data,
    so shared references are safe across concurrent submissions. Each
    replay still has its own ``_portfolio`` / ``_step_count`` /
    ``_trade_history`` / ``_done`` because those are reset in
    ``__init__`` and again in ``reset()``.
    """
    from openfinai_harbor.verifier.curated.task_loader import instantiate_task

    cfg = state.config
    spec = cfg.spec
    proto = state.handler_state[_HANDLER_KEY]["prototype"]
    fresh = instantiate_task(state.task_cls, default_config=spec.default_config)
    # load_data is idempotent on _data; reusing the prototype's caches
    # short-circuits the CSV load on the replay's first reset().
    fresh._data = proto._data
    fresh._ohlcv = getattr(proto, "_ohlcv", {})
    fresh._prices = getattr(proto, "_prices", {})
    fresh._min_len = getattr(proto, "_min_len", 0)
    return fresh


def _zero_action_for(task: Any) -> Any:
    """A no-op action for the (uniformly transactional) trading tasks.

    Used by the on_startup probe and as a defensive default when an
    incoming action doesn't match the schema (we still want a clean
    dispatch — a ``hold`` is a no-op the executor accepts).
    """
    return {"action": "hold"}


def _json_safe(value: Any) -> Any:
    """Coerce an observation / info payload to JSON-serialisable primitives.

    The offline trading observation is already plain floats/lists, but
    info dicts can carry numpy scalars (OHLCV slices) or nested
    execution-report dicts, so walk the structure once.
    """
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    return value


def _sweep_sessions(sessions: Dict[str, Dict[str, Any]], now: float) -> None:
    """Drop sessions idle past the TTL (best-effort GC on each call)."""
    stale = [
        sid
        for sid, s in sessions.items()
        if now - s["last_active"] > _SESSION_TTL_SEC
    ]
    for sid in stale:
        sessions.pop(sid, None)


def _do_session_reset(
    state, req: CuratedSessionResetRequest  # type: CuratedVerifierState
) -> CuratedSessionResetResponse:
    """Open a fresh interactive episode, return the first causal observation.

    The verifier holds the task instance; the agent only ever sees the
    observation at the current cursor, so it cannot read a future bar.
    """
    hs = state.handler_state[_HANDLER_KEY]
    sessions = hs.setdefault("sessions", {})
    now = time.time()
    _sweep_sessions(sessions, now)

    sid = req.session_id or uuid.uuid4().hex
    existing = sessions.get(sid)
    if existing is not None and existing["agent_id"] != req.agent_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="session_id is owned by a different agent",
        )

    # Per-agent cap: evict the agent's least-recently-active session if it
    # already holds the maximum (a same-id restart doesn't count).
    agent_sids = [
        s_id
        for s_id, s in sessions.items()
        if s["agent_id"] == req.agent_id and s_id != sid
    ]
    if len(agent_sids) >= _MAX_SESSIONS_PER_AGENT:
        oldest = min(agent_sids, key=lambda s_id: sessions[s_id]["last_active"])
        sessions.pop(oldest, None)

    replay = _build_replay_task(state)
    obs = replay.reset()
    sessions[sid] = {
        "agent_id": req.agent_id,
        "task": replay,
        "created": now,
        "last_active": now,
        "step": 0,
        "actions": [],
    }
    return CuratedSessionResetResponse(
        session_id=sid,
        observation=_json_safe(obs) if obs is not None else None,
        step=0,
        done=bool(getattr(replay, "_done", False)),
    )


def _do_session_step(
    state, req: CuratedSessionStepRequest  # type: CuratedVerifierState
) -> CuratedSessionStepResponse:
    """Advance one bar; on the terminal step, score server-side.

    Scoring stays on the verifier (reads the authoritative task's
    ``_trade_history``), so the agent can't tamper with its own reward.
    """
    hs = state.handler_state[_HANDLER_KEY]
    sessions = hs.setdefault("sessions", {})
    now = time.time()
    _sweep_sessions(sessions, now)

    entry = sessions.get(req.session_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="unknown or expired session_id; call /session/reset first",
        )
    if entry["agent_id"] != req.agent_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="session_id is owned by a different agent",
        )
    task = entry["task"]
    if getattr(task, "_done", False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="episode already finished; call /session/reset to restart",
        )

    obs, reward, done, info = task.step(req.action)
    entry["step"] += 1
    entry["actions"].append(req.action)
    entry["last_active"] = now

    resp = CuratedSessionStepResponse(
        session_id=req.session_id,
        observation=_json_safe(obs) if obs is not None else None,
        reward=float(reward),
        done=bool(done),
        step=entry["step"],
        info=_json_safe(info),
    )
    if done:
        scores = task.evaluate(entry["actions"])
        flat, weighted_total = coerce_scores(scores, HEADLINE_METRIC)
        flat["steps_executed"] = float(entry["step"])
        resp.scores = flat
        resp.weighted_total = weighted_total
        audit_log(
            state,
            {
                "ts_utc": registry.utc_now_iso(),
                "agent_id": req.agent_id,
                "submission_id": req.session_id,
                "ok": True,
                "scores": flat,
                "weighted_total": weighted_total,
                "steps_executed": entry["step"],
                "mode": "interactive_session",
            },
        )
        # The finished session is kept (task._done=True) so a stray re-step
        # gets a clear 409 rather than a confusing 404; the idle-TTL sweep and
        # the per-agent cap reclaim it.
    return resp


def register_routes(app: FastAPI, state, *, bearer) -> None:  # type: CuratedVerifierState
    """Mount held-out-safe reset/step endpoints with server-side scoring."""

    @app.post("/session/reset", response_model=CuratedSessionResetResponse)
    async def session_reset(
        req: CuratedSessionResetRequest,
        request: Request,
        token: str = Depends(bearer),
    ) -> Union[CuratedSessionResetResponse, JSONResponse]:
        # Rate-limit episode creation (the throttle point); per-step calls
        # below are bounded by the episode length so they aren't capped.
        allowed, retry_after = state.rate_limiter.check_and_record(token)
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": (
                        f"rate limit exceeded; retry after {retry_after:.1f}s"
                    )
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
        record_submit_heartbeat(state, req.agent_id)
        with state.score_lock:
            return _do_session_reset(state, req)

    @app.post("/session/step", response_model=CuratedSessionStepResponse)
    async def session_step(
        req: CuratedSessionStepRequest,
        request: Request,
        token: str = Depends(bearer),
    ) -> CuratedSessionStepResponse:
        # Steps are gated on an existing session and bounded by the episode,
        # so they aren't rate-limited; the heartbeat keeps the agent live
        # through a long episode.
        record_submit_heartbeat(state, req.agent_id)
        with state.score_lock:
            return _do_session_step(state, req)
