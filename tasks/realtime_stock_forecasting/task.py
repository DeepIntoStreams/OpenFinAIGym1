"""Deferred equity-price forecasting with Alpaca market data.

Predictions are resolved after the configured bar horizon. Requires
``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY``.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from openfinai_pipeline.benchmark.contracts import TaskMetadata
from openfinai_pipeline.realtime.data_providers.alpaca import AlpacaProvider
from openfinai_pipeline.realtime.ledger import PredictionLedger
from openfinai_pipeline.realtime.tasks.base_realtime_task import RealtimeForecastingTask

_RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
_DEFAULT_DB = _RESULTS_DIR / "predictions.db"


class RealtimeStockForecasting(RealtimeForecastingTask):
    """Realtime forecasting task for US equities.

    ``config`` selects the symbol universe, scored subset, data resolutions,
    forecast horizon, ledger path, and headline metric. Predictions outside
    the scored subset are rejected. ``provider`` and ``ledger`` are injectable.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        provider: Optional[Any] = None,
        ledger: Optional[PredictionLedger] = None,
    ) -> None:
        config = config or {}
        symbols = config.get("symbols", ["SPY"])
        horizon_bars = int(config.get("horizon_bars", 5))
        context_resolutions = config.get(
            "context_resolutions",
            [{"interval": "1m", "bars": 60}],
        )
        data_resolution = config.get("data_resolution")
        target_symbols = config.get("target_symbols")
        db_path = config.get("db_path", str(_DEFAULT_DB))
        headline_metric = str(config.get("headline_metric", "price_mape"))

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        if provider is None:
            provider = AlpacaProvider()
        if ledger is None:
            ledger = PredictionLedger(db_path)

        super().__init__(
            config=config,
            provider=provider,
            ledger=ledger,
            symbols=symbols,
            horizon_bars=horizon_bars,
            context_resolutions=context_resolutions,
            data_resolution=data_resolution,
            target_symbols=target_symbols,
            headline_metric=headline_metric,
        )

    def metadata(self) -> TaskMetadata:
        base = super().metadata()
        sym_label = "_".join(s.lower() for s in self._symbols)
        return TaskMetadata(
            task_id=f"realtime_stock_forecasting_{sym_label}",
            title=f"Realtime Stock Forecasting ({', '.join(self._symbols)}, Alpaca)",
            description=(
                "Streaming price forecasting on US equities via the Alpaca "
                "data API. Agent submits an absolute predicted_price for the "
                f"next {self._horizon_minutes}-minute horizon; direction is "
                "derived server-side. Ground truth is resolved after the "
                f"horizon elapses. Headline metric: {self._headline_metric}."
            ),
            interaction_model=base.interaction_model,
            task_type=base.task_type,
            data_requirements=base.data_requirements,
            tags=["stock", "forecasting", "realtime", "alpaca"],
            difficulty=base.difficulty,
            version="2.0.0",
        )
