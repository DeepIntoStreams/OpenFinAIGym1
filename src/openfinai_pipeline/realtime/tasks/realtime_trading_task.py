"""Realtime trading backed by one live-data buffer and a paper executor."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from data.knowledge_base.rewards import (
    MaxDrawdown,
    PnL,
    SharpeRatio,
    TradingReward,
    WinRate,
)
from openfinai_pipeline.benchmark.contracts import (
    TaskMetadata,
    TradingTask,
)
from openfinai_pipeline.settings import TradingConfig
from openfinai_pipeline.realtime.data_providers.base import (
    DataProvider,
    downsample_bars,
)
from openfinai_pipeline.realtime.execution import (
    BaseExecutor,
    OrderType,
    create_executor,
)
from openfinai_pipeline.realtime.market_data_buffer import MarketDataBuffer
from openfinai_pipeline.realtime.tasks.base_realtime_task import (
    _resolve_context_resolutions,
    _validate_target_symbols,
)

logger = logging.getLogger(__name__)


class RealtimeTradingTask(TradingTask):
    """Trade at the finest resolution and downsample coarser contexts.

    ``target_symbols`` limits evaluation, not trading. Alpaca mode uses the
    broker's cash instead of ``initial_capital``.
    """

    #: Rewards run during evaluation, not per tick.
    DEFAULT_TRADING_REWARDS: tuple[type[TradingReward], ...] = (
        PnL,
        SharpeRatio,
        MaxDrawdown,
        WinRate,
    )

    #: WS staleness limits before REST fallback.
    PRICE_STALENESS_TTL_MS: float = 500.0
    OB_STALENESS_TTL_MS: float = 1000.0

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        provider: DataProvider,
        symbols: list[str],
        trading_config: TradingConfig | None = None,
        context_resolutions: list[dict[str, Any]] | None = None,
        data_resolution: str | None = None,
        buffer_size: int = 500,
        max_steps: int = 0,
        target_symbols: list[str] | None = None,
        initial_capital: float = 100000.0,
    ) -> None:
        super().__init__(config)
        self._provider = provider
        self._symbols = list(symbols)
        if not self._symbols:
            raise ValueError("symbols must be non-empty")
        self._trading_config = trading_config or TradingConfig()
        self._max_steps = max_steps
        self._initial_capital = float(initial_capital)
        # Sidecars downsample the primary buffer; lookback spans the deepest one.
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
        # Context-only symbols may be traded but are excluded from evaluation.
        self._target_symbols = _validate_target_symbols(
            target_symbols, self._symbols
        )

        # Ensure the primary buffer spans every sidecar window.
        required_buffer = primary_lookback + max(data_resolution_bars, 100)
        effective_buffer_size = max(buffer_size, required_buffer)

        self._buffer = MarketDataBuffer(
            provider=provider,
            symbols=self._symbols,
            interval=data_resolution_interval,
            buffer_size=effective_buffer_size,
            backfill_bars=self._lookback_bars,
        )
        self._executor: BaseExecutor = create_executor(
            self._trading_config,
            provider.name,
            initial_capital=self._initial_capital,
        )

        self._step_count: int = 0
        self._done: bool = False
        self._trade_history: List[Dict[str, Any]] = []
        # Persistent pool fans out price and order-book REST calls per symbol.
        self._fetch_pool = ThreadPoolExecutor(
            max_workers=max(4, len(self._symbols) * 2),
            thread_name_prefix="rtt-fetch",
        )

    def metadata(self) -> TaskMetadata:
        sym_label = "_".join(self._symbols[:3])
        return TaskMetadata(
            task_id=f"realtime_trading_{sym_label}",
            title=f"Realtime Trading ({', '.join(self._symbols)})",
            description=(
                f"Realtime paper trading for {self._symbols} via {self._provider.name} "
                f"with {self._trading_config.execution_mode} execution."
            ),
            task_type="realtime",
            interaction_model="trading",
        )

    def load_data(self) -> Any:
        """Backfill, start streaming, and allow up to five seconds for warmup.

        A warmup timeout is nonfatal because REST supplies stale or cold data.
        """
        if self._data is not None:
            return self._data
        self._buffer.backfill()
        stream_thread = self._buffer.start_streaming_background()
        # WS-less providers return None and require no warmup wait.
        if stream_thread is not None:
            warm = self._buffer.wait_for_warmup(timeout=5.0)
            if not warm:
                logger.warning(
                    "WS warmup timeout (5s) for some symbols; hot path "
                    "will fall back to REST until WS catches up. "
                    "provider=%s symbols=%s",
                    self._provider.name,
                    self._symbols,
                )
        self._data = True
        return self._data

    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "keys": {
                "step": "int",
                "steps_remaining": "int (-1 when max_steps is unbounded)",
                "symbols": {
                    "per_symbol": {
                        "symbol": "str",
                        "price": "float",
                        "volume": (
                            "float | None — in-progress bar's running cumulative "
                            "volume (monotonically increasing within the minute "
                            "for Binance WS klines, reset at minute close). None "
                            "when the provider hasn't supplied volume yet."
                        ),
                        "timestamp": "str (ISO-8601; bar-boundary)",
                        "recent_bars": f"list[MarketSnapshot] (last {self._context_bars})",
                        "order_book": "OrderBookSnapshot | None",
                    },
                },
                "portfolio": (
                    "dict[cash (float | None if executor doesn't track cash "
                    "locally), reserved_cash, positions(dict[str,float]), "
                    "pnl(dict[str,float]), value(float | None), "
                    "pending_orders(list[PendingOrder])]"
                ),
            },
        }

    def get_action_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "keys": {
                "action": "'buy' | 'sell' | 'hold' | 'cancel' (required)",
                "symbol": "str (required for buy/sell)",
                "quantity": "float (> 0 for buy/sell, ignored for hold/cancel)",
                "order_type": (
                    "'market' | 'limit' | 'stop' | 'stop_limit' (default 'market')"
                ),
                "limit_price": "float (required for limit / stop_limit)",
                "stop_price": "float (required for stop / stop_limit)",
                "tif": "'ioc' | 'gtc' (default 'gtc'; ignored for market/hold)",
                "order_id": "str (required only for action='cancel')",
            },
            "batch": (
                "{'orders': [<single-order dict>, ...]} for multiple orders / "
                "cancels in one step"
            ),
            "order_types": list(OrderType.ALL),
            "time_in_force": ["ioc", "gtc"],
        }

    def reset(self) -> Any:
        """Reset executor and buffer; return initial observation."""
        self._executor.reset()
        self._step_count = 0
        self._done = False
        self._trade_history = []
        self._prev_pnl_per_sym = {}
        if self._data is None:
            self.load_data()
        # Seed the first observation while the stream starts.
        self._refresh_prices_from_provider()
        return self._get_market_observation()

    def _on_step_start(self) -> None:
        # Refresh once so execution and the following observation share prices.
        self._refresh_prices_from_provider()

    def _market_observation_block(self) -> Dict[str, Dict[str, Any]]:
        # Read only from the buffer; step() owns provider refreshes.
        sym_obs: Dict[str, Dict[str, Any]] = {}
        for symbol in self._symbols:
            latest = self._buffer.get_latest_price(symbol)
            recent = self._buffer.get_recent_bars(symbol, self._context_bars)
            order_book = self._buffer.get_order_book(symbol)

            entry: Dict[str, Any] = {
                "symbol": symbol,
                "price": latest.price if latest else None,
                "volume": latest.volume if latest else None,
                "timestamp": latest.timestamp.isoformat() if latest else None,
                "recent_bars": recent,
                "order_book": order_book,
            }
            if self._extra_resolutions:
                # Downsample one shared history to keep all resolutions aligned.
                by_interval: dict[str, list[Any]] = {self._interval: recent}
                full_history = self._buffer.get_recent_bars(
                    symbol, self._lookback_bars
                )
                for ex_interval, ex_bars in self._extra_resolutions:
                    by_interval[ex_interval] = downsample_bars(
                        full_history, ex_interval, ex_bars
                    )
                entry["recent_bars_by_interval"] = by_interval
            sym_obs[symbol] = entry
        return sym_obs

    def _execution_quotes(self) -> Dict[str, float]:
        # In-progress bars expose only scalar execution prices.
        return self._current_prices()

    def _market_timestamp(self) -> Any:
        return datetime.now(timezone.utc)

    def _episode_done(self) -> bool:
        return self._max_steps > 0 and self._step_count >= self._max_steps - 1

    def _steps_remaining(self) -> int:
        return self._max_steps - self._step_count if self._max_steps > 0 else -1

    def _current_prices(self) -> dict[str, float]:
        """Collect latest prices for all symbols."""
        prices: dict[str, float] = {}
        for s in self._symbols:
            lp = self._buffer.get_latest_price(s)
            if lp is not None:
                prices[s] = lp.price
        return prices

    def _refresh_prices_from_provider(self) -> None:
        """Fetch stale prices and books through the REST fallback.

        Results use the WS callbacks, and staleness is rechecked after I/O so a
        newer WS update is never overwritten.
        """
        if not self._symbols:
            return
        futures: list[tuple[str, str, Any]] = []
        has_ob_api = getattr(self._provider, "get_order_book", None) is not None
        for sym in self._symbols:
            if self._buffer.price_staleness_ms(sym) >= self.PRICE_STALENESS_TTL_MS:
                futures.append((
                    sym, "price",
                    self._fetch_pool.submit(
                        self._provider.get_current_price, sym
                    ),
                ))
            if (
                has_ob_api
                and self._buffer.ob_staleness_ms(sym) >= self.OB_STALENESS_TTL_MS
            ):
                futures.append((
                    sym, "ob",
                    self._fetch_pool.submit(self._provider.get_order_book, sym),
                ))
        if not futures:
            return
        for sym, kind, future in futures:
            try:
                result = future.result(timeout=15.0)
            except Exception:
                logger.debug(
                    "parallel %s fetch failed for %s", kind, sym, exc_info=True
                )
                continue
            if kind == "price":
                # Do not overwrite a WS update that arrived during the REST
                # call. TTL=0 remains the explicit always-REST mode.
                if (
                    self._buffer.price_staleness_ms(sym)
                    >= self.PRICE_STALENESS_TTL_MS
                ):
                    self._buffer.insert_bar(sym, result)
            else:  # "ob"
                if (
                    self._buffer.ob_staleness_ms(sym)
                    >= self.OB_STALENESS_TTL_MS
                ):
                    self._buffer._on_order_book(result)

    def close(self) -> None:
        """Shut down the per-task worker pool; safe to call repeatedly."""
        pool = getattr(self, "_fetch_pool", None)
        if pool is None:
            return
        pool.shutdown(wait=False)
        self._fetch_pool = None  # type: ignore[assignment]
