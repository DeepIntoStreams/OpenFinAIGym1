"""Common submit-handler envelope for curated verifier endpoints.

Every curated ``/submit/*`` endpoint shares the same lifecycle:

1. Rate-limit check (returns 429 with ``Retry-After`` if blocked).
2. Idempotent cache lookup keyed by ``submission_id``.
3. Heartbeat record (a ``/submit`` IS proof of life).
4. Score under ``state.score_lock`` with try/except + audit-on-error.
5. Build ``CuratedSubmitResponse`` from the closure's outcome.
6. Cache + LRU evict.
7. Audit-on-success.

Family-specific work (validation, scoring, audit detail) lives in a
sync ``score_fn`` closure passed by the handler. The closure either
returns a :class:`SubmitOutcome` describing the response, or raises
:class:`HTTPException` (validation errors propagate without auditing
as internal failures — they're caller bugs, not server bugs).
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, Union

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

from openfinai_harbor.verifier import registry
from openfinai_harbor.verifier._lifecycle import (
    audit_log,
    record_submit_heartbeat,
)
from openfinai_harbor.verifier.curated.models import CuratedSubmitResponse


@dataclass
class SubmitOutcome:
    """What the handler-specific score closure returns to the envelope.

    ``status="scored"`` is the synchronous-success case; ``"deferred"``
    is for realtime forecasting where ground truth resolves later via
    the resolver service. ``deferred`` carries session metadata in
    deferred mode and is None for scored submissions. ``audit_extras``
    is merged into the success-audit payload (the envelope already
    populates ts_utc / agent_id / submission_id / ok / elapsed_sec /
    scores / weighted_total).
    """

    status: Literal["scored", "deferred"]
    scores: Dict[str, float]
    weighted_total: float
    deferred: Optional[Dict[str, Any]] = None
    audit_extras: Dict[str, Any] = field(default_factory=dict)


ScoreFn = Callable[[], SubmitOutcome]


async def run_submit(
    *,
    state: Any,
    agent_id: str,
    submission_id: str,
    token: str,
    score_fn: ScoreFn,
    error_detail_prefix: str = "scoring failed",
) -> Union[CuratedSubmitResponse, JSONResponse]:
    """Run a curated submit endpoint with full lifecycle envelope.

    ``score_fn`` runs under ``state.score_lock``; it can do handler-
    specific validation (raise :class:`HTTPException` to abort cleanly
    without auditing as an internal error) or perform the actual scoring
    and return a :class:`SubmitOutcome`.

    Returns either:

    * :class:`CuratedSubmitResponse` — success path (including cached
      re-serves, where ``cached=True`` is set on the copy).
    * :class:`JSONResponse` (429) — rate-limit blocked.

    Any internal failure inside ``score_fn`` is audited and re-raised
    as a 500 :class:`HTTPException` with detail
    ``f"{error_detail_prefix}: {exc}"`` so handlers can give callers a
    helpful diagnosis (e.g. ``"trading replay failed: ..."`` vs
    ``"deferred ledger write failed: ..."``).
    """
    # 1. Rate-limit
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

    # 2. Idempotent cache check
    with state.state_lock:
        cached = state.submission_cache.get(submission_id)
        if cached is not None:
            return cached.model_copy(update={"cached": True})

    # 3. Heartbeat (a /submit IS proof of life)
    record_submit_heartbeat(state, agent_id)

    # 4. Score under lock, with audit-on-error
    t0 = time.time()
    try:
        with state.score_lock:
            outcome = score_fn()
    except HTTPException:
        # Validation errors propagate without being audited as internal
        # failures — they're caller bugs, not server bugs.
        raise
    except Exception as exc:
        elapsed = time.time() - t0
        tb = traceback.format_exc(limit=8)
        audit_log(
            state,
            {
                "ts_utc": registry.utc_now_iso(),
                "agent_id": agent_id,
                "submission_id": submission_id,
                "ok": False,
                "elapsed_sec": elapsed,
                "error": str(exc),
                "traceback": tb[-2000:],
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{error_detail_prefix}: {exc}",
        ) from exc
    elapsed = time.time() - t0

    # 5. Build response
    response = CuratedSubmitResponse(
        submission_id=submission_id,
        status=outcome.status,
        scores=outcome.scores,
        weighted_total=outcome.weighted_total,
        elapsed_sec=elapsed,
        cached=False,
        deferred=outcome.deferred,
    )

    # 6. Cache + LRU evict
    with state.state_lock:
        state.submission_cache[submission_id] = response
        while len(state.submission_cache) > state.config.submission_cache_size:
            state.submission_cache.popitem(last=False)

    # 7. Audit success
    audit_log(
        state,
        {
            "ts_utc": registry.utc_now_iso(),
            "agent_id": agent_id,
            "submission_id": submission_id,
            "ok": True,
            "elapsed_sec": elapsed,
            "scores": outcome.scores,
            "weighted_total": outcome.weighted_total,
            **outcome.audit_extras,
        },
    )
    return response


__all__ = ["SubmitOutcome", "run_submit"]
