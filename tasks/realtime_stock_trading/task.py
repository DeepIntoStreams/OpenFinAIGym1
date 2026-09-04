"""Realtime equity trading with Alpaca market data.

Requires ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY``. The internal mode
simulates execution locally; the Alpaca-paper mode submits orders to the
configured paper account.
"""

from typing import Any, Dict, Optional

from openfinai_pipeline.benchmark.contracts import TaskMetadata
from openfinai_pipeline.realtime.data_providers.alpaca import AlpacaProvider
from openfinai_pipeline.realtime.tasks.realtime_trading_task import RealtimeTradingTask
from openfinai_pipeline.settings import TradingConfig


class RealtimeStockTrading(RealtimeTradingTask):
    """Paper-trading task for US equities.

    ``config`` selects symbols, market-data resolutions, episode length,
    execution mode, costs, and the symbols included in scoring. ``provider``
    may replace the Alpaca data provider.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        provider: Optional[Any] = None,
    ) -> None:
        config = config or {}
        symbols = config.get("symbols", ["SPY"])
        max_steps = int(config.get("max_steps", 0))
        execution_mode = config.get("execution_mode", "internal_paper")
        context_resolutions = config.get(
            "context_resolutions",
            [{"interval": "1m", "bars": 60}],
        )
        data_resolution = config.get("data_resolution")
        target_symbols = config.get("target_symbols")
        initial_capital = float(config.get("initial_capital", 100000.0))

        trading_config = TradingConfig(
            slippage_pct=float(config.get("slippage_pct", 0.001)),
            transaction_cost_pct=float(config.get("transaction_cost_pct", 0.0)),
            execution_mode=execution_mode,
        )
        if provider is None:
            provider = AlpacaProvider()

        super().__init__(
            config=config,
            provider=provider,
            symbols=symbols,
            trading_config=trading_config,
            context_resolutions=context_resolutions,
            data_resolution=data_resolution,
            max_steps=max_steps,
            target_symbols=target_symbols,
            initial_capital=initial_capital,
        )

    def metadata(self) -> TaskMetadata:
        base = super().metadata()
        sym_label = "_".join(s.lower() for s in self._symbols)
        return TaskMetadata(
            task_id=f"realtime_stock_trading_{sym_label}",
            title=f"Realtime Stock Trading ({', '.join(self._symbols)}, Alpaca)",
            description=(
                "Realtime paper trading on US equities via the Alpaca data API. "
                "Agent submits buy/sell/hold actions with quantities; rewards "
                "are immediate mark-to-market PnL."
            ),
            interaction_model=base.interaction_model,
            task_type=base.task_type,
            data_requirements=base.data_requirements,
            tags=["stock", "trading", "realtime", "alpaca"],
            difficulty=base.difficulty,
            version="1.0.0",
        )
