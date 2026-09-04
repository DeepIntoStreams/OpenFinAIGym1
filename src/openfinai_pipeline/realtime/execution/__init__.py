"""Execution backends for realtime paper trading.

Use :func:`create_executor` to instantiate the correct backend based on
:class:`TradingConfig.execution_mode`.
"""

from openfinai_pipeline.settings import TradingConfig
from openfinai_pipeline.realtime.execution.base import (
    ActionVerb,
    BaseExecutor,
    CancelResult,
    ExecutionReport,
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
    Rejection,
    RejectionCode,
    SubmitResult,
    TickResult,
    TimeInForce,
)
from openfinai_pipeline.realtime.execution.simulated import SimulatedExecutor

__all__ = [
    "ActionVerb",
    "BaseExecutor",
    "CancelResult",
    "ExecutionReport",
    "OrderIntent",
    "OrderStatus",
    "OrderType",
    "PendingOrder",
    "Rejection",
    "RejectionCode",
    "SimulatedExecutor",
    "SubmitResult",
    "TickResult",
    "TimeInForce",
    "create_executor",
]


def create_executor(
    config: TradingConfig,
    provider_name: str = "binance",
    *,
    initial_capital: float = 100000.0,
) -> BaseExecutor:
    """Instantiate the correct executor based on *config.execution_mode*.

    Raises :class:`ValueError` if ``"alpaca_paper"`` is requested but
    the provider is not ``"alpaca"``.
    """
    if config.execution_mode == "alpaca_paper":
        if provider_name != "alpaca":
            raise ValueError(
                f"alpaca_paper execution mode requires the alpaca provider, "
                f"got {provider_name!r}"
            )
        from openfinai_pipeline.realtime.execution.alpaca_paper import AlpacaPaperExecutor

        return AlpacaPaperExecutor(flatten_on_start=config.flatten_on_start)
    return SimulatedExecutor(config, initial_capital=initial_capital)
