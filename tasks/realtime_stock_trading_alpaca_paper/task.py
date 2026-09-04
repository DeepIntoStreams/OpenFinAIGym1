"""Equity trading through Alpaca's paper-trading API.

Requires ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY``. Use a dedicated paper
account: the executor assumes exclusive control, and ``reset()`` cancels all
open orders and closes all positions in that account.
"""

from typing import Any, Dict, Optional

from openfinai_pipeline.benchmark.contracts import TaskMetadata
from openfinai_pipeline.realtime.data_providers.alpaca import AlpacaProvider
from openfinai_pipeline.realtime.tasks.realtime_trading_task import RealtimeTradingTask
from openfinai_pipeline.settings import TradingConfig


class RealtimeStockTradingAlpacaPaper(RealtimeTradingTask):
    """Alpaca-paper task with a fixed ``alpaca_paper`` execution mode.

    ``config`` selects symbols, market-data resolutions, episode length, and
    the symbols included in scoring. Trades outside the scoring set still
    affect the paper account. ``provider`` must be Alpaca-compatible.
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
        context_resolutions = config.get(
            "context_resolutions",
            [{"interval": "1m", "bars": 60}],
        )
        data_resolution = config.get("data_resolution")
        target_symbols = config.get("target_symbols")
        # The executor reads cash from Alpaca; this value is not its balance.
        initial_capital = float(config.get("initial_capital", 100000.0))

        trading_config = TradingConfig(
            slippage_pct=0.0,
            transaction_cost_pct=0.0,
            execution_mode="alpaca_paper",
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
            task_id=f"realtime_stock_trading_alpaca_paper_{sym_label}",
            title=(
                f"Realtime Stock Trading via Alpaca Paper API "
                f"({', '.join(self._symbols)})"
            ),
            description=(
                "Realtime paper trading on US equities. Orders are submitted to "
                "Alpaca's paper-trading environment at paper-api.alpaca.markets "
                "and filled against Alpaca's simulated order book; PnL is read "
                "from account positions. Requires ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables."
            ),
            interaction_model=base.interaction_model,
            task_type=base.task_type,
            data_requirements=base.data_requirements,
            tags=["stock", "trading", "realtime", "alpaca", "alpaca_paper"],
            difficulty=base.difficulty,
            version="1.0.0",
        )
