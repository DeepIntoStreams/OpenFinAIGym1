"""Per-task verifier API for held-out scoring.

The server caches ground truth, accepts authenticated prediction submissions,
and returns aggregate scores only. Shared lifecycle code provides registration,
heartbeats, health, authentication, rate limits, and audits.
"""

from __future__ import annotations

import base64
import importlib.util
import inspect
import io
import logging
import sys
import threading
import time
import traceback
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from openfinai_harbor.verifier import registry
from openfinai_harbor.verifier._lifecycle import (
    AgentRecord,
    audit_log,
    bearer_dependency,
    mount_lifecycle_routes,
    record_submit_heartbeat,
    start_lifecycle_thread,
)
from openfinai_harbor.verifier.auth import RateLimiter
from openfinai_harbor.verifier.models import (
    InfoResponse,
    SubmitRequest,
    SubmitResponse,
)


logger = logging.getLogger("openfinai_harbor.verifier")


@dataclass
class VerifierConfig:
    """Per-task configuration captured at startup.

    Resolved by ``__main__.py`` from CLI flags + ``task.toml``. Captures
    everything the server needs to operate; not mutated after startup.
    """

    task_id: str
    task_dir: Path                  # tasks/generated/<scope>/<task_id>/
    data_dir: Path                  # task_dir/environment/data/
    eval_data_dir: Path             # task_dir/environment/eval-data/
    evaluator_path: Path            # data_dir/evaluator.py
    interaction_model: str          # 'forecasting' | 'generative' | ...
    has_held_out_test_gt: bool
    state_dir: Path                 # data/run_output/verifier/<task_id>/
    run_dir: Path                   # state_dir/runs/<utc_ts>/
    token: str
    despawn_grace_min: float
    heartbeat_timeout_sec: float = 90.0
    liveness_poll_sec: float = 30.0
    max_requests_per_minute: int = 60
    submission_cache_size: int = 64


@dataclass
class VerifierState:
    """Mutable per-process state. Guarded by ``state_lock``."""

    config: VerifierConfig
    agents: Dict[str, AgentRecord] = field(default_factory=dict)
    submission_cache: "OrderedDict[str, SubmitResponse]" = field(
        default_factory=OrderedDict
    )
    rate_limiter: RateLimiter = field(init=False)
    last_idle_at: Optional[float] = None  # set when agents falls to zero
    started_at: float = field(default_factory=time.time)
    started_at_iso: str = field(default_factory=registry.utc_now_iso)
    n_total_submissions: int = 0
    # Cached evaluator + ground truth.
    evaluator: Any = None
    evaluator_module: Any = None
    cached_gt: Any = None              # numpy array, dict-of-arrays, or None
    reward_names: List[str] = field(default_factory=list)
    expects_fake_emb: bool = False
    # Threading.
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    score_lock: threading.Lock = field(default_factory=threading.Lock)
    audit_lock: threading.Lock = field(default_factory=threading.Lock)
    lifecycle_stop: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self.rate_limiter = RateLimiter(
            max_requests=self.config.max_requests_per_minute,
            window_sec=60.0,
        )


def _load_evaluator_module(evaluator_path: Path) -> Tuple[Any, Any]:
    """Load the assembled evaluator module + return (module, evaluator_instance).

    Adds the data_dir to sys.path so ``task_rewards.py`` (sibling of
    ``evaluator.py``) is importable when the evaluator's
    ``importlib.util.spec_from_file_location`` falls back to it.
    Mirrors the load pattern used by ``run_evaluation_explicit.py``.
    """
    if not evaluator_path.exists():
        raise FileNotFoundError(f"evaluator not found: {evaluator_path}")
    data_dir = evaluator_path.parent
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))

    mod_name = f"_verifier_evaluator_{abs(hash(str(evaluator_path)))}"
    spec = importlib.util.spec_from_file_location(mod_name, str(evaluator_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    evaluator_cls = None
    for attr in dir(module):
        obj = getattr(module, attr, None)
        if not isinstance(obj, type):
            continue
        if obj.__module__ != module.__name__:
            continue
        if (
            hasattr(obj, "REWARD_NAMES")
            and hasattr(obj, "score")
            and obj.__name__ != "BaseEvaluator"
        ):
            evaluator_cls = obj
            break
    if evaluator_cls is None:
        raise RuntimeError(f"no evaluator class in {evaluator_path}")
    return module, evaluator_cls()


def _preload_ground_truth(
    *,
    evaluator_module: Any,
    evaluator: Any,
    data_dir: Path,
    eval_data_dir: Path,
    has_held_out_test_gt: bool,
) -> Any:
    """Best-effort: pre-load held-out gt at startup.

    Uses the assembled evaluator's own ``_load_reference_data`` helper if
    available (preserves dispatch consistency with the file-IO path).
    Failure here is fatal at startup — better to surface "eval-data
    missing" immediately than at first submission.

    Args:
        evaluator_module: The imported evaluator module; consulted for an
            optional ``_load_reference_data`` helper.
        evaluator: The evaluator instance (held for symmetry with
            file-IO path; not currently dereferenced here).
        data_dir: Bundled data directory passed to the helper.
        eval_data_dir: Verifier-owned eval-data directory passed to the
            helper.
        has_held_out_test_gt: When False (e.g. ``shape="reference"``
            unconditional generative), short-circuit to ``None`` so
            ``score()`` can synthesise its placeholder.

    Returns:
        The pre-loaded ground truth (ndarray, dict of ndarrays, or
        custom shape per evaluator), or ``None`` when no held-out gt
        exists or the evaluator exposes no loader helper.
    """
    if not has_held_out_test_gt:
        return None

    loader = getattr(evaluator_module, "_load_reference_data", None)
    if loader is None:
        # Evaluator doesn't expose a loader helper; let score() lazy-load.
        return None

    try:
        sig = inspect.signature(loader)
        kwargs: Dict[str, Any] = {}
        # Loader signatures vary across generations; pass only accepted kwargs.
        for pname in sig.parameters:
            if pname == "data_dir":
                kwargs["data_dir"] = str(data_dir)
            elif pname == "eval_data_dir":
                kwargs["eval_data_dir"] = str(eval_data_dir)
        return loader(**kwargs)
    except TypeError:
        try:
            return loader(str(eval_data_dir))
        except Exception:
            return None
    except Exception as exc:
        raise RuntimeError(
            f"failed to pre-load held-out ground truth from {eval_data_dir}: {exc}"
        ) from exc


def _detect_expects_fake_emb(evaluator: Any) -> bool:
    """Best-effort: does any reward use the ``fake_emb`` canonical name?"""
    specs = getattr(evaluator, "_REWARD_SPECS", None) or getattr(
        type(evaluator), "_REWARD_SPECS", None
    )
    if specs is None:
        return False
    for spec in specs:
        for key in ("forward_ctx_args", "init_ctx_args"):
            args = spec.get(key) if isinstance(spec, dict) else None
            if args and "fake_emb" in args:
                return True
    return False


def _decode_b64_array(b64: str) -> np.ndarray:
    """Decode a base64-encoded numpy ``.npy`` byte string back to ndarray."""
    raw = base64.b64decode(b64.encode("ascii"))
    buf = io.BytesIO(raw)
    return np.load(buf, allow_pickle=False)


def _do_submit(state: VerifierState, req: SubmitRequest) -> SubmitResponse:
    cfg = state.config
    # Idempotent retry: composite (agent_id, submission_id) key so two agents
    # that reuse the same submission_id string don't get each other's score.
    cache_key = (getattr(req, "agent_id", "") or "", req.submission_id)
    with state.state_lock:
        cached = state.submission_cache.get(cache_key)
        if cached is not None:
            cached_copy = cached.model_copy(update={"cached": True})
            return cached_copy

    # Decode predictions.
    try:
        predictions = _decode_b64_array(req.predictions_b64)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"invalid predictions_b64: {exc}",
        ) from exc
    fake_emb: Optional[np.ndarray] = None
    if req.fake_emb_b64 is not None:
        try:
            fake_emb = _decode_b64_array(req.fake_emb_b64)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"invalid fake_emb_b64: {exc}",
            ) from exc

    record_submit_heartbeat(state, req.agent_id)

    # Global lock: evaluator metric instances aren't thread-safe (e.g.
    # _ensure_metrics mutates dict on first call). Serialization cost is
    # negligible at our 10-concurrent-agents-max scale.
    score_kwargs: Dict[str, Any] = {
        "predictions": predictions,
        "ground_truth": state.cached_gt,
        "weights": None,
        "reward_output": None,
        "data_dir": str(cfg.data_dir),
        "eval_data_dir": str(cfg.eval_data_dir),
    }
    if fake_emb is not None:
        score_kwargs["fake_emb"] = fake_emb
    for k, v in (req.extra or {}).items():
        score_kwargs.setdefault(k, v)

    t0 = time.time()
    try:
        with state.score_lock:
            scores = state.evaluator.score(**score_kwargs)
    except Exception as exc:
        elapsed = time.time() - t0
        tb = traceback.format_exc(limit=8)
        audit_log(
            state,
            {
                "ts_utc": registry.utc_now_iso(),
                "agent_id": req.agent_id,
                "submission_id": req.submission_id,
                "ok": False,
                "elapsed_sec": elapsed,
                "error": str(exc),
                "traceback": tb[-2000:],
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"evaluator.score raised: {exc}",
        ) from exc
    elapsed = time.time() - t0

    # Preserve the evaluator's canonical reward in both the flat score map and
    # weighted_total for older clients.
    raw_scores = dict(scores) if isinstance(scores, dict) else {}
    weighted_total = 0.0
    flat: Dict[str, float] = {}
    for name, value in raw_scores.items():
        if name.startswith("_"):
            continue  # internal fields like _errors
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            coerced = 0.0
        flat[name] = coerced
        if name == "reward":
            weighted_total = coerced
    response = SubmitResponse(
        submission_id=req.submission_id,
        scores=flat,
        weighted_total=weighted_total,
        elapsed_sec=elapsed,
        cached=False,
    )

    # Cache (bounded LRU). Same composite key as the lookup above.
    with state.state_lock:
        state.submission_cache[cache_key] = response
        while len(state.submission_cache) > cfg.submission_cache_size:
            state.submission_cache.popitem(last=False)

    audit_log(
        state,
        {
            "ts_utc": registry.utc_now_iso(),
            "agent_id": req.agent_id,
            "submission_id": req.submission_id,
            "ok": True,
            "elapsed_sec": elapsed,
            "scores": flat,
            "weighted_total": weighted_total,
            "errors": raw_scores.get("_errors"),
        },
    )
    return response


def _do_info(state: VerifierState) -> InfoResponse:
    cfg = state.config
    with state.state_lock:
        n_active = len(state.agents)
    return InfoResponse(
        task_id=cfg.task_id,
        interaction_model=cfg.interaction_model,
        reward_names=list(state.reward_names),
        has_held_out_test_gt=cfg.has_held_out_test_gt,
        expects_fake_emb=state.expects_fake_emb,
        n_active_agents=n_active,
        started_at_utc=state.started_at_iso,
        despawn_grace_min=cfg.despawn_grace_min,
    )


def create_app(config: VerifierConfig) -> FastAPI:
    """Build a FastAPI app for one task.

    Initialises evaluator + cached gt synchronously so any startup
    failure surfaces immediately (uvicorn refuses to bind, exit non-zero).
    """
    state = VerifierState(config=config)

    # Pre-warm at construction (NOT lifespan) so a broken bundle fails before
    # uvicorn binds; clients then see connection-refused, surfaced cleanly.
    state.evaluator_module, state.evaluator = _load_evaluator_module(
        config.evaluator_path
    )
    state.reward_names = list(getattr(state.evaluator, "REWARD_NAMES", []) or [])
    state.cached_gt = _preload_ground_truth(
        evaluator_module=state.evaluator_module,
        evaluator=state.evaluator,
        data_dir=config.data_dir,
        eval_data_dir=config.eval_data_dir,
        has_held_out_test_gt=config.has_held_out_test_gt,
    )
    state.expects_fake_emb = _detect_expects_fake_emb(state.evaluator)

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        thread = start_lifecycle_thread(
            state, app, logger=logger, log_label="verifier"
        )
        logger.info(
            "verifier ready: task_id=%s reward_names=%s expects_fake_emb=%s",
            config.task_id,
            state.reward_names,
            state.expects_fake_emb,
        )
        try:
            yield
        finally:
            state.lifecycle_stop.set()
            thread.join(timeout=2.0)
            registry.remove_url_file(config.state_dir)
            logger.info("verifier shut down: task_id=%s", config.task_id)

    app = FastAPI(
        title=f"OpenFinGym Verifier ({config.task_id})",
        description=(
            "Per-task verifier RPC service. Holds held-out test ground "
            "truth in memory and serves prediction submissions."
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
                "n_active_agents": len(state.agents),
                "n_total_submissions": state.n_total_submissions,
            }

    @app.get("/info", response_model=InfoResponse)
    async def info(_: str = Depends(bearer)) -> InfoResponse:
        return _do_info(state)

    mount_lifecycle_routes(app, state, bearer=bearer, logger=logger)

    @app.post("/submit", response_model=SubmitResponse)
    async def submit(
        req: SubmitRequest,
        request: Request,
        token: str = Depends(bearer),
    ) -> SubmitResponse:
        # Rate-limit before doing any work.
        allowed, retry_after = state.rate_limiter.check_and_record(token)
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": (
                        f"rate limit exceeded "
                        f"({state.config.max_requests_per_minute} req / 60s); "
                        f"retry after {retry_after:.1f}s"
                    ),
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
        # Offload the CPU-bound scoring path to a worker thread so the
        # asyncio event loop stays free to service /healthz, /heartbeat,
        # and other agents' /register concurrently. Without this, an
        # evaluator that takes several seconds (e.g. the fBM ONND metric)
        # would silently fail the run_trial heartbeat poll even though
        # the verifier is alive and scoring is progressing.
        return await run_in_threadpool(_do_submit, state, req)

    return app
