"""Realtime crypto paper trading with public Binance market data."""

from typing import Any, Dict, Optional

from openfinai_pipeline.benchmark.contracts import TaskMetadata
from openfinai_pipeline.settings import TradingConfig
from openfinai_pipeline.realtime.data_providers.binance import BinanceProvider
from openfinai_pipeline.realtime.tasks.realtime_trading_task import RealtimeTradingTask


class RealtimeCryptoTrading(RealtimeTradingTask):
    """Paper-trading task for Binance crypto markets.

    ``config`` selects symbols, market-data resolutions, episode length,
    costs, and the symbols included in scoring. ``provider`` may replace the
    Binance data provider.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        provider: Optional[Any] = None,
    ) -> None:
        config = config or {}
        symbols = config.get("symbols", ["BTCUSDT"])
        max_steps = int(config.get("max_steps", 0))
        context_resolutions = config.get(
            "context_resolutions",
            [{"interval": "1m", "bars": 60}],
        )
        data_resolution = config.get("data_resolution")
        target_symbols = config.get("target_symbols")
        initial_capital = float(config.get("initial_capital", 100000.0))

        # Binance data is paired only with the in-process paper executor.
        trading_config = TradingConfig(
            slippage_pct=float(config.get("slippage_pct", 0.001)),
            transaction_cost_pct=float(config.get("transaction_cost_pct", 0.0)),
            execution_mode="internal_paper",
        )
        if provider is None:
            provider = BinanceProvider()

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
            task_id=f"realtime_crypto_trading_{sym_label}",
            title=f"Realtime Crypto Trading ({', '.join(self._symbols)}, Binance)",
            description=(
                "Realtime paper trading on Binance crypto markets. Agent submits "
                "buy/sell/hold actions with quantities; rewards are immediate "
                "mark-to-market PnL. Uses the free Binance public API."
            ),
            interaction_model=base.interaction_model,
            task_type=base.task_type,
            data_requirements=base.data_requirements,
            tags=["crypto", "trading", "realtime", "binance"],
            difficulty=base.difficulty,
            version="1.0.0",
        )
