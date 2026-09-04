"""RealtimeForecastingTask — ForecastingTask subclass for realtime / deferred-eval.

This task is fundamentally a forecasting task, not a trading task: the
agent submits directional predictions (``long``/``short`` + confidence)
and ground truth is resolved asynchronously by the :class:`ResolverService`
after the prediction horizon has elapsed. No orders are executed, no
portfolio state is maintained.

Data flows as Python objects in-memory (no file I/O on the hot path).

* ``reset()``   — fetches a fresh market snapshot via the data provider.
* ``step()``    — records the agent's prediction in the ledger and returns
                  the next market snapshot.  The step reward is always 0
                  because ground truth is not yet available.
* ``evaluate()`` — returns deferred-status metadata.  Actual rewards are
                   computed later by the :class:`ResolverService`.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from openfinai_pipeline.benchmark.contracts import ForecastingTask, TaskMetadata
from openfinai_pipeline.realtime.data_providers.base import (
    DataProvider,
    downsample_bars,
    interval_to_seconds,
    interval_to_timedelta,
    resolve_at_for_horizon,
)
from openfinai_pipeline.realtime.ledger import PredictionLedger

logger = logging.getLogger(__name__)


#: Headline-metric whitelist for ``RealtimeForecastingTask``. Covers the
#: full price/return/directional panel produced by
#: :data:`ALL_TRADING_REWARDS`. Both ``directional_accuracy``
#: (handler-side spelling) and ``direction_accuracy`` (the reward
#: class's ``.name`` attribute) are accepted; the handler resolves the
#: per-spelling mismatch when reading the headline value.
_VALID_REALTIME_HEADLINE: tuple[str, ...] = (
    "price_mape",
    "price_mse",
    "price_rmse",
    "price_mae",
    "price_r2",
    "price_pearson",
    "return_mae",
    "return_mse",
    "return_rmse",
    "direction_accuracy",
    "directional_accuracy",
    "pnl",
    "sharpe_ratio",
    "win_rate",
    "max_drawdown",
)


def _validate_target_symbols(
    target_symbols: list[str] | None,
    symbols: list[str],
) -> list[str]:
    """Validate ``target_symbols`` is a non-empty subset of ``symbols``.

    Returns ``symbols`` (a copy) when ``target_symbols`` is ``None`` or
    empty so legacy configs behave unchanged. Raises ``ValueError`` if
    the user passes a set that is not a subset of ``symbols`` — a typo
    here is silent failure if we accept it (the resolver would never
    find ground truth for the rogue symbol).
    """
    if not target_symbols:
        return list(symbols)
    targets = list(target_symbols)
    extras = [s for s in targets if s not in symbols]
    if extras:
        raise ValueError(
            f"target_symbols must be a subset of symbols. "
            f"Unknown: {extras}; symbols: {symbols}"
        )
    if not targets:
        raise ValueError("target_symbols must be non-empty when provided")
    return targets


def _resolve_context_resolutions(
    context_resolutions: list[dict[str, Any]] | None,
    data_resolution: str | None,
) -> tuple[str, int, tuple[tuple[str, int], ...], int]:
    """Validate resolutions and return primary, sidecars, and lookback.

    Intervals must be unique and the selected primary must be the finest.
    Sidecars are downsampled from it, and the returned lookback spans the
    deepest requested sidecar window. Resolution does not pace agent steps.
    """
    if not context_resolutions:
        raise ValueError(
            "context_resolutions must be a non-empty list of "
            "{'interval': str, 'bars': positive int} entries"
        )
    parsed: list[tuple[str, int]] = []
    seen: set[str] = set()
    for i, entry in enumerate(context_resolutions):
        if not isinstance(entry, dict):
            raise ValueError(
                f"context_resolutions[{i}] must be a dict with 'interval' "
                f"and 'bars'; got {type(entry).__name__}"
            )
        interval = entry.get("interval")
        bars = entry.get("bars")
        if not isinstance(interval, str) or not interval:
            raise ValueError(
                f"context_resolutions[{i}] missing string 'interval'"
            )
        if not isinstance(bars, int) or bars <= 0:
            raise ValueError(
                f"context_resolutions[{i}].bars must be a positive int; "
                f"got {bars!r}"
            )
        if interval in seen:
            raise ValueError(
                f"context_resolutions[{i}].interval={interval!r} appears "
                "more than once; intervals must be unique"
            )
        seen.add(interval)
        parsed.append((interval, int(bars)))

    if data_resolution is None:
        if len(parsed) > 1:
            raise ValueError(
                f"data_resolution must be specified when context_resolutions "
                f"has more than one entry; got intervals "
                f"{[i for i, _ in parsed]}"
            )
        data_resolution_interval, data_resolution_bars = parsed[0]
    else:
        if not isinstance(data_resolution, str) or not data_resolution:
            raise ValueError(
                f"data_resolution must be a non-empty string; got "
                f"{data_resolution!r}"
            )
        match = next(
            ((iv, bb) for iv, bb in parsed if iv == data_resolution),
            None,
        )
        if match is None:
            raise ValueError(
                f"data_resolution={data_resolution!r} does not match any "
                f"entry in context_resolutions; available intervals: "
                f"{[i for i, _ in parsed]}"
            )
        data_resolution_interval, data_resolution_bars = match

    extras = tuple(
        (iv, bb) for iv, bb in parsed if iv != data_resolution_interval
    )

    # Sidecars can only be downsampled from a finer primary interval.
    data_resolution_seconds = interval_to_seconds(data_resolution_interval)
    if data_resolution_seconds <= 0:
        raise ValueError(
            f"data_resolution={data_resolution_interval!r} parsed to "
            "non-positive seconds"
        )
    for ex_interval, _ in extras:
        ex_seconds = interval_to_seconds(ex_interval)
        if ex_seconds <= data_resolution_seconds:
            raise ValueError(
                f"data_resolution={data_resolution_interval!r} "
                f"({data_resolution_seconds}s) is not finer than sidecar "
                f"interval {ex_interval!r} ({ex_seconds}s); sidecars must "
                "be coarser than the data resolution because they are "
                "derived by downsampling from the primary buffer."
            )

    # Size the primary buffer to span the deepest sidecar window.
    primary_lookback = data_resolution_bars
    for ex_interval, ex_bars in extras:
        ex_seconds = interval_to_seconds(ex_interval)
        ratio = ex_seconds // data_resolution_seconds
        # safe: data_resolution_seconds > 0, ex_seconds > data_resolution_seconds
        primary_lookback = max(primary_lookback, ex_bars * ratio)

    return (
        data_resolution_interval,
        data_resolution_bars,
        extras,
        primary_lookback,
    )


class RealtimeForecastingTask(ForecastingTask):
    """Submit independent realtime predictions for deferred evaluation.

    The primary resolution defines the target-bar horizon and buffer cadence;
    coarser contexts are downsampled from it. The resolver later fetches exit
    prices and scores only ``target_symbols``. Agent step frequency is not
    paced by the data resolution.
    """

    batch_mode: bool = False

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        provider: DataProvider,
        ledger: PredictionLedger,
        symbols: list[str],
        horizon_bars: int,
        context_resolutions: list[dict[str, Any]] | None = None,
        data_resolution: str | None = None,
        target_symbols: list[str] | None = None,
        headline_metric: str = "price_mape",
    ) -> None:
        super().__init__(config)
        self._provider = provider
        self._ledger = ledger
        self._symbols = list(symbols)
        if not self._symbols:
            raise ValueError("symbols must be non-empty")
        self._horizon_bars = int(horizon_bars)
        if self._horizon_bars < 1:
            raise ValueError(
                f"horizon_bars must be >= 1; got {self._horizon_bars!r}"
            )
        if headline_metric not in _VALID_REALTIME_HEADLINE:
            raise ValueError(
                f"headline_metric={headline_metric!r} is not in the "
                f"realtime forecasting panel. Valid options: "
                f"{sorted(_VALID_REALTIME_HEADLINE)}"
            )
        self._headline_metric = headline_metric
        # Sidecars are downsampled from the primary buffer at obs time;
        # _lookback_bars auto-scales to span the deepest sidecar.
        if context_resolutions is None:
            context_resolutions = [{"interval": "1m", "bars": 60}]
        (
            data_resolution_interval,
            data_resolution_bars,
            extras,
            primary_lookback,
        ) = _resolve_context_resolutions(context_resolutions, data_resolution)
        self._interval = data_resolution_interval
        self._context_bars = data_resolution_bars
        self._lookback_bars = primary_lookback
        self._extra_resolutions: tuple[tuple[str, int], ...] = extras
        # Horizon is specified in bars of the primary (finest) resolution,
        # mirroring offline's forecast_horizon_bars. The wall-clock minute
        # span is derived for the ledger's horizon_minutes column + display.
        self._horizon_minutes = int(
            self._horizon_bars * interval_to_seconds(self._interval) / 60
        )
        # target_symbols defaults to all inputs; validated as a non-
        # empty subset so a typo can't silently disable scoring.
        self._target_symbols = _validate_target_symbols(
            target_symbols, self._symbols
        )
        self._session_id: str = ""
        # Shadows ForecastingTask._predictions (raw actions) with rich
        # ledger records carrying entry_price / resolve_at / ledger_id.
        self._predictions: list[dict[str, Any]] = []
        self._step_count: int = 0

    # ForecastingTask / BaseTask contract

    def metadata(self) -> TaskMetadata:
        sym_label = "_".join(self._symbols)
        return TaskMetadata(
            task_id=f"realtime_forecasting_{sym_label}_{self._horizon_minutes}m",
            title=f"Realtime Forecasting ({', '.join(self._symbols)}, "
                  f"{self._horizon_minutes}m horizon)",
            description=(
                f"Predict price direction for {self._symbols} over a "
                f"{self._horizon_minutes}-minute horizon using real-time market data."
            ),
            task_type="realtime",
            interaction_model="forecasting",
        )

    def get_features(self) -> Any:
        """Return the current market observation as the feature set.

        Realtime forecasting has no static feature matrix; each prediction is
        made against the latest market snapshot. Callers that take the
        batch forecasting path (``batch_mode=True``) will get a single
        snapshot to predict from.
        """
        if self._data is None:
            self.load_data()
        return self._build_observation()

    def get_ground_truth(self) -> Any:
        """Ground truth is deferred for realtime forecasting.

        Returns ``None`` because the actual outcomes are resolved
        asynchronously by :class:`ResolverService` after the prediction
        horizon elapses. The :meth:`evaluate` override returns a deferred
        status dict instead of delegating to :meth:`predict_and_evaluate`.
        """
        return None

    def load_data(self) -> Any:
        """Pre-fetch recent bar history for each symbol at the step
        resolution.

        The primer window is ``primary_lookback × data_resolution`` —
        auto-scaled by :func:`_resolve_context_resolutions` to span the
        deepest sidecar window so the primary history can be
        downsampled into every sidecar's ``recent_bars_by_interval``
        entry without fetching extra streams. Sidecars are NOT
        independently fetched; they are derived on demand via
        :func:`downsample_bars` in :meth:`_build_observation`.
        """
        if self._data is not None:
            return self._data
        now = datetime.now(timezone.utc)
        backfill = interval_to_timedelta(self._interval, self._lookback_bars)
        self._data = {}
        for symbol in self._symbols:
            try:
                bars = self._provider.get_bars(
                    symbol,
                    self._interval,
                    start=now - backfill,
                    end=now,
                )
                self._data[symbol] = bars
            except Exception:
                logger.warning("Failed to load history for %s", symbol, exc_info=True)
                self._data[symbol] = []
        return self._data

    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "per_symbol": {
                "symbol": "str",
                "price": "float",
                "timestamp": "str (ISO-8601)",
                "recent_bars": f"list[MarketSnapshot] (last {self._context_bars})",
                "history_available": "bool",
            },
        }

    def get_action_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "keys": {
                "symbol": "str (required)",
                "predicted_price": (
                    "float > 0 (recommended) — absolute predicted future "
                    "price; direction is derived from "
                    "sign(predicted_price - entry_price)"
                ),
                "direction": (
                    "'long' | 'short' (optional / legacy) — supply this "
                    "instead of (or alongside) predicted_price; if both "
                    "are given, they must agree"
                ),
                "predicted_return": "float (optional, signed magnitude)",
                "confidence": "float 0-1 (optional)",
            },
            "constraints": (
                "at least one of {predicted_price, direction, "
                "predicted_return} must be supplied per submission"
            ),
        }

    def reset(self) -> Any:
        """Start a new prediction session; return initial market snapshots."""
        self._session_id = str(uuid4())
        self._predictions = []
        self._step_count = 0
        if self._data is None:
            self.load_data()
        return self._build_observation()

    def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        """Record agent prediction, return next market snapshot.

        ``action`` must include ``symbol`` plus at least one of
        ``predicted_price`` (recommended) or ``direction`` (legacy
        path). When both are supplied, ``direction`` must agree with
        ``sign(predicted_price - entry_price)`` or :class:`ValueError`
        is raised — silently overriding either field would mask agent
        bugs. Predictions whose ``symbol`` is not in
        :attr:`_target_symbols` are dropped with a warning (no ledger
        write, no resolver work, ``info`` carries
        ``"dropped_off_target": True``). The agent still advances a
        step so the gym loop terminates on ``max_steps``.
        """
        symbol = action["symbol"]
        if symbol not in self._target_symbols:
            logger.warning(
                "Prediction for %r dropped: symbol not in target_symbols=%s "
                "(input symbols=%s). Configure target_symbols to scope "
                "scoring to a subset of inputs.",
                symbol,
                self._target_symbols,
                self._symbols,
            )
            self._step_count += 1
            return self._build_observation(), 0.0, False, {
                "dropped_off_target": True,
                "symbol": symbol,
            }
        snapshot = self._provider.get_current_price(symbol)
        now = datetime.now(timezone.utc)

        predicted_price = action.get("predicted_price")
        direction = action.get("direction")
        if predicted_price is not None:
            predicted_price = float(predicted_price)
            if predicted_price <= 0:
                raise ValueError(
                    f"predicted_price must be > 0; got {predicted_price!r}"
                )
            derived = (
                "long" if predicted_price >= float(snapshot.price) else "short"
            )
            if direction is None:
                direction = derived
            elif direction != derived:
                raise ValueError(
                    f"direction={direction!r} contradicts "
                    f"sign(predicted_price - entry_price)={derived!r} "
                    f"(predicted_price={predicted_price}, "
                    f"entry_price={snapshot.price})"
                )
        elif direction is None:
            raise ValueError(
                "action must supply at least one of predicted_price "
                "or direction"
            )

        resolve_at = resolve_at_for_horizon(
            now, self._interval, self._horizon_bars
        )
        pred_record: dict[str, Any] = {
            "session_id": self._session_id,
            "provider": self._provider.name,
            "symbol": symbol,
            "direction": direction,
            "predicted_price": predicted_price,
            "predicted_return": action.get("predicted_return"),
            "confidence": action.get("confidence"),
            "entry_price": snapshot.price,
            "horizon_minutes": self._horizon_minutes,
            "resolution_interval": self._interval,
            "submitted_at": now,
            "resolve_at": resolve_at,
        }
        pred_id = self._ledger.submit(pred_record)
        self._predictions.append(pred_record)
        self._step_count += 1

        info: Dict[str, Any] = {
            "prediction_id": pred_id,
            "resolve_at": pred_record["resolve_at"].isoformat(),
            "entry_price": snapshot.price,
            "predicted_price": predicted_price,
            "direction": direction,
        }
        obs = self._build_observation()
        return obs, 0.0, False, info  # reward deferred

    def evaluate(self, agent_actions: List[Any], **kwargs: Any) -> Dict[str, float]:
        """Return deferred-status metadata (rewards computed by resolver)."""
        return {
            "status_deferred": 1.0,
            "n_predictions": float(len(self._predictions)),
            "session_id_hash": float(hash(self._session_id) % 10**6),
        }

    # helpers

    def _build_observation(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {}
        for symbol in self._symbols:
            try:
                snap = self._provider.get_current_price(symbol)
            except Exception:
                logger.warning("get_current_price failed for %s", symbol, exc_info=True)
                snap = None
            history = self._data.get(symbol, []) if self._data else []
            # Append current snapshot so recent_bars stays fresh across steps
            if snap is not None:
                history.append(snap)
                if self._data is not None and symbol in self._data:
                    self._data[symbol] = history
            sym_obs: Dict[str, Any] = {
                "symbol": symbol,
                "price": snap.price if snap else None,
                "timestamp": snap.timestamp.isoformat() if snap else None,
                "recent_bars": history[-self._context_bars :],
                "history_available": bool(history),
            }
            if self._extra_resolutions:
                # Multi-resolution view: sidecars downsample from the
                # primary history — no extra fetch, no staleness.
                by_interval: dict[str, list[Any]] = {
                    self._interval: sym_obs["recent_bars"],
                }
                for ex_interval, ex_bars in self._extra_resolutions:
                    by_interval[ex_interval] = downsample_bars(
                        history, ex_interval, ex_bars
                    )
                sym_obs["recent_bars_by_interval"] = by_interval
            obs[symbol] = sym_obs
        return obs

    @property
    def session_id(self) -> str:
        return self._session_id
