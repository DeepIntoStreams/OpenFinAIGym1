"""Record realtime forecasts for deferred scoring.

The verifier captures entry prices, creates pending ledger rows, and returns
session metadata. Agent-provided prices cannot set the entry price, and ground
truth is unavailable until the resolver processes the horizon.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
from uuid import uuid4

from openfinai_pipeline.realtime.data_providers.base import (
    resolve_at_for_horizon,
)

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from openfinai_harbor.verifier.curated._envelope import (
    SubmitOutcome,
    run_submit,
)
from openfinai_harbor.verifier.curated.models import (
    CuratedDeferredSubmitRequest,
    CuratedSubmitResponse,
)


logger = logging.getLogger(__name__)


# Full metric panel advertised via /info; price metrics NaN-skip when
# predicted_price is missing (direction-only submissions).
REWARD_NAMES = [
    "price_mse",
    "price_rmse",
    "price_mae",
    "price_mape",
    "price_r2",
    "price_pearson",
    "return_mse",
    "return_rmse",
    "return_mae",
    "direction_accuracy",
]

#: Fallback when the task class doesn't expose ``_headline_metric``;
#: written to reward.json after the resolver runs (0.0 at submit time).
HEADLINE_METRIC = "price_mape"


_HANDLER_KEY = "realtime_forecasting"


def on_startup(state) -> None:  # type: CuratedVerifierState
    """Build the cached task + open a long-lived ledger connection.

    Curated realtime forecasting tasks (``RealtimeCryptoForecasting``,
    ``RealtimeStockForecasting``) hold their own ``PredictionLedger`` +
    ``DataProvider`` instances after construction. We instantiate once,
    extract those handles, and reuse them across submissions — no per-
    submission task state to reset because the handler bypasses
    ``task.step()`` and writes to the ledger directly.

    Failures here propagate up through ``create_curated_app`` and abort
    the spawn — better to surface a broken ledger path or unreachable
    data provider before uvicorn binds the port.
    """
    from openfinai_harbor.verifier.curated.task_loader import instantiate_task

    cfg = state.config
    spec = cfg.spec

    # SQLite predictions DB lives inside the verifier's run_dir: keeps
    # artifacts together and prevents inter-trial bleed (each spawn gets
    # its own run_dir + DB). The toml's `db_path` is only a fallback for
    # task instantiation outside the framework (unit tests).
    trial_db_path = cfg.run_dir / "predictions.db"
    trial_db_path.parent.mkdir(parents=True, exist_ok=True)
    default_config = dict(spec.default_config or {})
    default_config["db_path"] = str(trial_db_path)
    task = instantiate_task(state.task_cls, default_config=default_config)

    # Handler is tightly coupled to the curated task class: the
    # deferred-via-ledger wire contract is family-scoped.
    ledger = getattr(task, "_ledger", None)
    provider = getattr(task, "_provider", None)
    # Horizon is specified in bars of the task's primary (finest) resolution;
    # the bar interval drives where the resolver reads the exit price. The
    # derived minute span is kept only for the ledger's NOT NULL column and
    # diagnostics.
    horizon_bars = int(getattr(task, "_horizon_bars", 1))
    resolution_interval = str(getattr(task, "_interval", "1m"))
    horizon_minutes = int(getattr(task, "_horizon_minutes", 0))
    # Target subset preferred so the ledger doesn't accumulate predictions
    # the resolver can't score; full symbols list is the fallback.
    symbols = list(
        getattr(task, "_target_symbols", None)
        or getattr(task, "_symbols", [])
    )

    if ledger is None:
        raise RuntimeError(
            f"realtime forecasting task {type(task).__name__} did not expose "
            "a _ledger attribute; the curated handler requires direct ledger "
            "access (bypassing task.step) for the deferred submission flow"
        )

    handler_state: Dict[str, Any] = {
        "task": task,
        "ledger": ledger,
        "provider": provider,
        "horizon_bars": horizon_bars,
        "resolution_interval": resolution_interval,
        "horizon_minutes": horizon_minutes,
        "symbols": symbols,
        "ledger_path": str(getattr(ledger, "_db_path", "")),
        "provider_name": getattr(provider, "name", "") if provider else "",
        # Per-task headline; falls back to module-level HEADLINE_METRIC
        # when the task didn't expose one (direction-only configs).
        "headline_metric": str(
            getattr(task, "_headline_metric", HEADLINE_METRIC)
        ),
    }
    state.handler_state[_HANDLER_KEY] = handler_state
    state.reward_names = list(REWARD_NAMES)


def register_routes(app: FastAPI, state, *, bearer) -> None:  # type: CuratedVerifierState
    """Mount ``POST /submit/predictions_async`` for deferred forecasting."""

    @app.post("/submit/predictions_async", response_model=CuratedSubmitResponse)
    async def submit_predictions_async(
        req: CuratedDeferredSubmitRequest,
        request: Request,
        token: str = Depends(bearer),
    ) -> Union[CuratedSubmitResponse, JSONResponse]:
        def _score() -> SubmitOutcome:
            h = state.handler_state[_HANDLER_KEY]
            ledger = h["ledger"]
            provider = h["provider"]
            horizon_bars = h["horizon_bars"]
            resolution_interval = h["resolution_interval"]
            horizon_minutes = h["horizon_minutes"]
            valid_symbols = set(h["symbols"])
            provider_name = h["provider_name"]

            session_id = str(uuid4())
            pred_ids: list[str] = []
            resolve_ats: list[datetime] = []

            for pred in req.predictions:
                if valid_symbols and pred.symbol not in valid_symbols:
                    # Reject symbols outside the configured set: silently
                    # ledgering them would leave them unresolvable.
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"symbol {pred.symbol!r} not in this task's "
                            f"configured set {sorted(valid_symbols)}"
                        ),
                    )

                submitted_at = (
                    datetime.fromisoformat(pred.submitted_at)
                    if pred.submitted_at
                    else datetime.now(timezone.utc)
                )
                if submitted_at.tzinfo is None:
                    submitted_at = submitted_at.replace(tzinfo=timezone.utc)
                # Bar-aligned: resolve on the close of the bar `horizon_bars`
                # ahead of the bar containing submission, at the task's
                # data_resolution granularity (mirrors offline shift(-N)).
                resolve_at = resolve_at_for_horizon(
                    submitted_at, resolution_interval, horizon_bars
                )

                # Entry price is ALWAYS captured server-side at
                # ledger.submit() time. The wire model has no entry_price
                # field; smuggled values via `extra` are ignored. This
                # enforces the "no fake prices in curated tasks" invariant
                # for forecasting bundles (trading bundles eliminate fake
                # prices by moving execution into the agent container).
                if provider is None:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"task has no data provider to fetch entry "
                            f"price for {pred.symbol}; this is a server-"
                            "side configuration bug, not an agent error"
                        ),
                    )
                snap = provider.get_current_price(pred.symbol)
                entry_price = float(snap.price)

                # Derive direction from predicted_price. Wire schema
                # guarantees at least one of {predicted_price, direction};
                # when both arrive they must agree, and inconsistent pairs
                # are rejected here since the server holds the
                # authoritative entry_price.
                derived_dir: Optional[str] = None
                if pred.predicted_price is not None:
                    derived_dir = (
                        "long"
                        if float(pred.predicted_price) >= entry_price
                        else "short"
                    )
                if pred.direction is not None and derived_dir is not None:
                    if pred.direction != derived_dir:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"direction={pred.direction!r} contradicts "
                                f"sign(predicted_price - entry_price)="
                                f"{derived_dir!r} for {pred.symbol} "
                                f"(predicted_price={pred.predicted_price}, "
                                f"entry_price={entry_price})"
                            ),
                        )
                    final_direction = pred.direction
                elif pred.direction is not None:
                    final_direction = pred.direction
                else:
                    final_direction = derived_dir  # guaranteed non-None by validator

                pred_id = ledger.submit(
                    {
                        "session_id": session_id,
                        "provider": provider_name,
                        "symbol": pred.symbol,
                        "horizon_minutes": horizon_minutes,
                        "resolution_interval": resolution_interval,
                        "submitted_at": submitted_at,
                        "resolve_at": resolve_at,
                        "direction": final_direction,
                        "predicted_price": pred.predicted_price,
                        "entry_price": float(entry_price),
                        "snapshot": {},
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
                "horizon_minutes": horizon_minutes,
                "resolve_at_first": (
                    min(resolve_ats).isoformat() if resolve_ats else None
                ),
                "resolve_at_last": (
                    max(resolve_ats).isoformat() if resolve_ats else None
                ),
                "prediction_ids": pred_ids,
            }

            # Empty `scores` + 0.0 weighted_total => "no headline yet"
            # in harbor's reward.json; the resolve_deferred CLI overwrites
            # both once gt resolves.
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
            error_detail_prefix="deferred ledger write failed",
        )
