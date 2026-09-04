"""Expose active Polymarket events and defer probability scoring.

The verifier captures entry prices and absolute resolution times server-side,
then stores pending predictions in the ledger. The resolver later supplies
outcomes; disputed markets are excluded upstream.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Union
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from openfinai_harbor.verifier.curated._envelope import (
    SubmitOutcome,
    run_submit,
)
from openfinai_harbor.verifier.curated.models import (
    CuratedEventSubmitRequest,
    CuratedSubmitResponse,
)


logger = logging.getLogger(__name__)


# Full event-reward panel advertised via /info; populated by the resolver.
REWARD_NAMES = [
    "brier_score",
    "log_loss",
    "expected_calibration_error",
]


# Default headline metric (overridable per-task via task._headline_metric).
HEADLINE_METRIC = "brier_score"


_HANDLER_KEY = "realtime_polymarket"


def on_startup(state) -> None:  # type: CuratedVerifierState
    """Build the cached task, ledger handle, and discovered universe.

    The polymarket task class calls ``provider.discover_active_markets``
    inside ``__init__``, so by the time ``on_startup`` runs the universe
    is already materialised on the task's ``_target_symbols`` /
    ``_market_resolve_at`` / ``_market_metadata`` attributes. We just
    snapshot those into the handler state for the request handlers to
    read.

    Failures here propagate up through ``create_curated_app`` and abort
    the spawn — better to surface a broken Polymarket API or empty
    universe before uvicorn binds the port.
    """
    from openfinai_harbor.verifier.curated.task_loader import instantiate_task

    cfg = state.config
    spec = cfg.spec

    # SQLite predictions DB lives inside the verifier's run_dir to keep
    # artifacts together and prevent inter-trial bleed. The toml's
    # `db_path` is only a fallback for task instantiation outside the
    # framework (unit tests).
    trial_db_path = cfg.run_dir / "predictions.db"
    trial_db_path.parent.mkdir(parents=True, exist_ok=True)
    default_config = dict(spec.default_config or {})
    default_config["db_path"] = str(trial_db_path)
    task = instantiate_task(state.task_cls, default_config=default_config)

    ledger = getattr(task, "_ledger", None)
    provider = getattr(task, "_provider", None)
    target_symbols = list(getattr(task, "_target_symbols", []))
    market_resolve_at: dict[str, datetime] = dict(
        getattr(task, "_market_resolve_at", {})
    )
    market_metadata: dict[str, dict[str, Any]] = dict(
        getattr(task, "_market_metadata", {})
    )

    if ledger is None:
        raise RuntimeError(
            f"realtime polymarket task {type(task).__name__} did not expose "
            "a _ledger attribute; the curated handler requires direct ledger "
            "access (bypassing task.step) for the deferred submission flow"
        )
    if provider is None:
        raise RuntimeError(
            f"realtime polymarket task {type(task).__name__} did not expose "
            "a _provider attribute; the handler needs the provider to "
            "capture entry_price server-side at submission time"
        )
    if not target_symbols:
        # Empty universe is a soft failure: verifier still spawns so
        # operators can hit GET /info, but /markets/active returns [] and
        # any submission 422s.
        logger.warning(
            "realtime polymarket task %s discovered ZERO active markets; "
            "the agent will see an empty universe (check discovery filters "
            "in task.toml [curated.default_config.discovery])",
            type(task).__name__,
        )

    handler_state: Dict[str, Any] = {
        "task": task,
        "ledger": ledger,
        "provider": provider,
        "target_symbols": target_symbols,
        "market_resolve_at": market_resolve_at,
        "market_metadata": market_metadata,
        "ledger_path": str(getattr(ledger, "_db_path", "")),
        "provider_name": getattr(provider, "name", "polymarket"),
        "headline_metric": str(
            getattr(task, "_headline_metric", HEADLINE_METRIC)
        ),
    }
    state.handler_state[_HANDLER_KEY] = handler_state
    state.reward_names = list(REWARD_NAMES)


def _market_payload(symbol: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Project a discovery payload to the JSON shape sent to the agent.

    The internal payload (from ``PolymarketProvider``) carries a ``raw``
    field with the full upstream record; we strip that out for the
    agent-facing response since it's verbose and contains fields not
    documented in the task contract. The agent should only depend on
    the fields enumerated here.
    """
    resolve_at = meta.get("resolution_at")
    return {
        "symbol": symbol,
        "question": meta.get("question", ""),
        "description": meta.get("description", ""),
        "resolution_at": resolve_at.isoformat()
        if isinstance(resolve_at, datetime)
        else None,
        "current_yes_price": meta.get("current_price"),
        "outcomes": list(meta.get("outcomes") or []),
        "best_bid": meta.get("best_bid"),
        "best_ask": meta.get("best_ask"),
        "spread": meta.get("spread"),
        "volume_24h": meta.get("volume_24h"),
        "liquidity": meta.get("liquidity"),
        "tags": list(meta.get("tags") or []),
        "categories": list(meta.get("categories") or []),
        "slug": meta.get("slug", ""),
    }


def register_routes(app: FastAPI, state, *, bearer) -> None:  # type: CuratedVerifierState
    """Mount ``GET /markets/active`` + ``POST /submit/event_predictions_async``."""

    @app.get("/markets/active")
    async def get_active_markets(
        request: Request,
        token: str = Depends(bearer),
    ) -> List[Dict[str, Any]]:
        """Return the trial's discovered Polymarket universe.

        Each entry carries the full agent-facing market context the
        agent needs to forecast intelligently. The list is frozen for
        the trial — re-calling this endpoint returns the same set.
        """
        h = state.handler_state[_HANDLER_KEY]
        target_symbols: list[str] = h["target_symbols"]
        market_metadata: dict[str, dict[str, Any]] = h["market_metadata"]
        return [
            _market_payload(sym, market_metadata[sym])
            for sym in target_symbols
            if sym in market_metadata
        ]

    @app.post("/submit/event_predictions_async", response_model=CuratedSubmitResponse)
    async def submit_event_predictions_async(
        req: CuratedEventSubmitRequest,
        request: Request,
        token: str = Depends(bearer),
    ) -> Union[CuratedSubmitResponse, JSONResponse]:
        def _score() -> SubmitOutcome:
            h = state.handler_state[_HANDLER_KEY]
            ledger = h["ledger"]
            provider = h["provider"]
            valid_symbols = set(h["target_symbols"])
            market_resolve_at: dict[str, datetime] = h["market_resolve_at"]
            market_metadata: dict[str, dict[str, Any]] = h["market_metadata"]
            provider_name = h["provider_name"]

            session_id = str(uuid4())
            pred_ids: list[str] = []
            resolve_ats: list[datetime] = []

            for pred in req.predictions:
                if pred.symbol not in valid_symbols:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"symbol {pred.symbol!r} not in this trial's "
                            f"discovered universe ({len(valid_symbols)} markets); "
                            "fetch GET /markets/active first to see the universe"
                        ),
                    )

                resolve_at = market_resolve_at.get(pred.symbol)
                if resolve_at is None:
                    # Should be impossible given the symbol-set check
                    # above, but guard defensively.
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"server-side: no resolve_at known for "
                            f"{pred.symbol!r}; this is a discovery bug"
                        ),
                    )

                submitted_at = (
                    datetime.fromisoformat(pred.submitted_at)
                    if pred.submitted_at
                    else datetime.now(timezone.utc)
                )
                if submitted_at.tzinfo is None:
                    submitted_at = submitted_at.replace(tzinfo=timezone.utc)

                # Capture entry_price (current YES price) server-side.
                # Same anti-fake-price discipline as realtime_forecasting:
                # the agent never sees / supplies entry_price.
                try:
                    snap = provider.get_current_price(pred.symbol)
                    entry_price = float(snap.price)
                except Exception as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            f"upstream Polymarket fetch failed for "
                            f"{pred.symbol!r}: {exc}"
                        ),
                    )

                # Events have absolute resolve_at, but the ledger schema
                # requires horizon_minutes NOT NULL; populate with the
                # minute-delta for diagnostic value.
                horizon_minutes = max(
                    0,
                    int((resolve_at - submitted_at).total_seconds() // 60),
                )

                meta = market_metadata.get(pred.symbol, {})
                end_date_raw = meta.get("resolution_at")
                end_date_iso = (
                    end_date_raw.isoformat()
                    if isinstance(end_date_raw, datetime)
                    else None
                )

                pred_id = ledger.submit(
                    {
                        "session_id": session_id,
                        "provider": provider_name,
                        "symbol": pred.symbol,
                        "horizon_minutes": horizon_minutes,
                        "submitted_at": submitted_at,
                        "resolve_at": resolve_at,  # per-market absolute time
                        # `direction` NOT NULL in schema; always "long YES"
                        # by construction (probability is the YES-side prob).
                        "direction": "long",
                        # YES probability stashed in `predicted_price`
                        # (semantic re-use). The resolver detects event-shape
                        # rows and routes through EventReward, not TradingReward.
                        "predicted_price": float(pred.predicted_yes_probability),
                        "entry_price": entry_price,
                        "snapshot": {
                            "event_prediction": True,
                            "predicted_yes_probability": float(
                                pred.predicted_yes_probability
                            ),
                            # Snapshot market context so inspection tools
                            # need no Polymarket re-fetch. Cache-hit only.
                            "question": meta.get("question"),
                            "categories": list(meta.get("categories") or []),
                            "end_date_iso": end_date_iso,
                        },
                        "trial_dir": req.trial_dir,
                    }
                )
                pred_ids.append(pred_id)
                resolve_ats.append(resolve_at)

            deferred_meta = {
                "session_id": session_id,
                "n_predictions": len(pred_ids),
                "ledger_path": h["ledger_path"],
                "provider": provider_name,
                "resolve_at_first": (
                    min(resolve_ats).isoformat() if resolve_ats else None
                ),
                "resolve_at_last": (
                    max(resolve_ats).isoformat() if resolve_ats else None
                ),
                "prediction_ids": pred_ids,
            }

            return SubmitOutcome(
                status="deferred",
                scores={},
                weighted_total=0.0,
                deferred=deferred_meta,
                audit_extras={
                    "session_id": session_id,
                    "n_predictions": len(pred_ids),
                    "ledger_path": h["ledger_path"],
                },
            )

        return await run_submit(
            state=state,
            agent_id=req.agent_id,
            submission_id=req.submission_id,
            token=token,
            score_fn=_score,
            error_detail_prefix="deferred event-ledger write failed",
        )
