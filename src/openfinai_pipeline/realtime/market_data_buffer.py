"""MarketDataBuffer -- unified REST backfill + WebSocket real-time data.

Maintains a rolling window of :class:`MarketSnapshot` bars per symbol,
with timestamp-based deduplication so that REST historical bars and
WebSocket realtime bars merge seamlessly.

Only used by :class:`RealtimeTradingTask`.  Realtime forecasting and offline
tasks do not need this component.
"""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from openfinai_pipeline.realtime.data_providers.base import (
    DataProvider,
    MarketSnapshot,
    OrderBookSnapshot,
    interval_to_timedelta,
)

logger = logging.getLogger(__name__)


class MarketDataBuffer:
    """Rolling bar buffer fed by REST backfill and optional WebSocket stream.

    Args:
        provider: A :class:`DataProvider` (REST) or
            :class:`StreamingDataProvider` (REST + WebSocket).
        symbols: Symbols to track.
        interval: Bar interval for backfill and WebSocket subscription
            (e.g. ``"1m"``).
        buffer_size: Maximum bars retained per symbol. Oldest bars are
            evicted when the limit is exceeded.
        backfill_bars: Number of most-recent bars to fetch via REST on
            :meth:`backfill`.
    """

    def __init__(
        self,
        provider: DataProvider,
        symbols: list[str],
        interval: str = "1m",
        buffer_size: int = 500,
        backfill_bars: int = 200,
    ) -> None:
        self._provider = provider
        self._symbols = list(symbols)
        self._interval = interval
        self._buffer_size = buffer_size
        self._backfill_bars = backfill_bars

        self._lock = threading.Lock()
        # symbol -> OrderedDict[timestamp_key, MarketSnapshot]
        self._bars: dict[str, OrderedDict[str, MarketSnapshot]] = {
            s: OrderedDict() for s in symbols
        }
        self._order_books: dict[str, OrderBookSnapshot | None] = {
            s: None for s in symbols
        }
        # Local receipt times drive cache staleness; they are not exchange
        # timestamps. ``perf_counter`` avoids coarse Windows monotonic-clock
        # resolution, and 0.0 denotes a value that has never been received.
        self._bar_recv_perf: dict[str, float] = {s: 0.0 for s in symbols}
        self._ob_recv_perf: dict[str, float] = {s: 0.0 for s in symbols}
        self._streaming: bool = False
        # Warmup completes after each symbol receives its first WS bar; REST
        # remains the fallback until then.
        self._warmup_events: dict[str, threading.Event] = {
            s: threading.Event() for s in symbols
        }

    # REST backfill

    def backfill(self) -> None:
        """Fetch recent historical bars via REST and populate the buffer.

        The window size is ``backfill_bars × interval`` so the wall-clock
        duration scales correctly with the bar resolution (a 200-bar
        backfill at 1m covers 200 minutes, at 1h covers 200 hours, etc.).

        Per-symbol ``get_bars`` calls run in parallel via a transient
        thread pool. Each symbol's REST call is independent (different
        endpoint params, different responses); the buffer write happens
        under ``_lock`` so cross-symbol concurrency is safe. The
        previously-sequential ``O(N_symbols × RTT)`` wall-clock cost
        collapses to ``O(RTT)`` for the slowest single fetch.
        """
        if not self._symbols:
            return
        now = datetime.now(timezone.utc)
        backfill_window = interval_to_timedelta(self._interval, self._backfill_bars)

        def _fetch_one(symbol: str) -> tuple[str, list[Any] | None, Exception | None]:
            try:
                bars = self._provider.get_bars(
                    symbol,
                    self._interval,
                    start=now - backfill_window,
                    end=now,
                )
                return symbol, bars, None
            except Exception as exc:
                return symbol, None, exc

        # Transient pool — backfill runs once per episode, so the
        # per-call setup cost (~50-200 µs) is negligible compared to
        # the network savings on multi-symbol tasks.
        with ThreadPoolExecutor(
            max_workers=max(2, len(self._symbols)),
            thread_name_prefix="mdb-backfill",
        ) as pool:
            futures = [
                pool.submit(_fetch_one, sym) for sym in self._symbols
            ]
            for future in futures:
                symbol, bars, err = future.result()
                if err is not None:
                    logger.warning("Backfill failed for %s", symbol, exc_info=err)
                    continue
                if not bars:
                    continue
                with self._lock:
                    for bar in bars:
                        self._insert_bar_unlocked(symbol, bar)

    # Internal helpers

    @staticmethod
    def _ts_key(ts: datetime) -> str:
        """Deterministic string key for deduplication."""
        return ts.isoformat()

    def _insert_bar_unlocked(self, symbol: str, bar: MarketSnapshot) -> None:
        """Insert bar into buffer (caller holds ``_lock``)."""
        buf = self._bars[symbol]
        key = self._ts_key(bar.timestamp)
        # Overwrite if same timestamp (dedup: WS update replaces REST bar)
        buf[key] = bar
        # Move to end so iteration order stays chronological
        buf.move_to_end(key)
        # Evict oldest entries if over capacity
        while len(buf) > self._buffer_size:
            buf.popitem(last=False)
        # Record receipt time for the staleness API. Backfill calls reach
        # here too; that's fine — when the gym loop starts there'll have
        # been a recent insertion either way.
        self._bar_recv_perf[symbol] = time.perf_counter()

    def insert_bar(self, symbol: str, bar: MarketSnapshot) -> None:
        """Thread-safe bar insertion (used by WebSocket callbacks)."""
        with self._lock:
            self._insert_bar_unlocked(symbol, bar)

    # WebSocket callbacks

    def _on_bar(self, bar: MarketSnapshot) -> None:
        """Callback invoked by WebSocket bar subscription."""
        self.insert_bar(bar.symbol, bar)
        # Mark the symbol as warmed up so `wait_for_warmup` can unblock.
        # Safe to call repeatedly — `Event.set()` is idempotent.
        event = self._warmup_events.get(bar.symbol)
        if event is not None:
            event.set()

    def _on_order_book(self, ob: OrderBookSnapshot) -> None:
        """Callback invoked by WebSocket order-book subscription."""
        with self._lock:
            self._order_books[ob.symbol] = ob
            self._ob_recv_perf[ob.symbol] = time.perf_counter()
        # An order-book push also satisfies warmup — the symbol's WS
        # stream is alive, even if the bar feed hasn't pushed yet.
        event = self._warmup_events.get(ob.symbol)
        if event is not None:
            event.set()

    # Streaming lifecycle

    #: Reconnect parameters for the WS supervision loop. The backoff
    #: doubles after each failed attempt up to ``WS_RECONNECT_MAX_S``
    #: (so 1s → 2s → 4s → 8s → 16s → 30s steady-state retry interval).
    #: Generous enough not to hammer the exchange after a brief blip,
    #: tight enough that a healthy reconnect lands well within the
    #: per-step REST fallback's TTL window.
    WS_RECONNECT_INITIAL_S: float = 1.0
    WS_RECONNECT_MAX_S: float = 30.0

    async def _subscribe_with_reconnect(
        self,
        kind: str,
        subscribe_coro_factory: Any,
    ) -> None:
        """Run a single WS subscription in an exponential-backoff loop.

        ``subscribe_coro_factory`` is a zero-arg callable that returns
        a fresh subscription coroutine each call. Once the subscription
        coroutine resolves (clean close) or raises, we log, sleep with
        exponential backoff, and re-invoke the factory — unless
        :meth:`stop_streaming` has flipped ``self._streaming`` to False,
        in which case we return.

        ``kind`` is a human-readable label used purely in log lines
        (e.g. ``"bars/BTCUSDT"`` or ``"ob/AAPL"``).
        """
        backoff = self.WS_RECONNECT_INITIAL_S
        while self._streaming:
            try:
                await subscribe_coro_factory()
                # Clean close — providers don't normally return; if one
                # does (e.g. server-closed the socket), treat it as a
                # blip and reconnect. Reset the backoff because the
                # connection *did* establish successfully.
                backoff = self.WS_RECONNECT_INITIAL_S
                logger.info(
                    "[%s] subscription returned cleanly; reconnecting",
                    kind,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "[%s] subscription error: %s — reconnecting in %.1fs",
                    kind,
                    exc,
                    backoff,
                )
            if not self._streaming:
                return
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(backoff * 2.0, self.WS_RECONNECT_MAX_S)

    def stop_streaming(self) -> None:
        """Signal the reconnect loops to exit at their next checkpoint.

        Idempotent. Does not actively cancel an in-flight ``recv()`` —
        the loops poll ``self._streaming`` between reconnect attempts
        and at the top of each iteration, so shutdown is observed
        within at most one ``recv`` timeout / one backoff interval.
        Caller is responsible for actually cancelling the asyncio task
        if a faster teardown is required.
        """
        self._streaming = False

    async def start_streaming(self) -> None:
        """Start WebSocket subscriptions for all symbols.

        Each per-symbol subscription is wrapped in
        :meth:`_subscribe_with_reconnect`, so transient disconnects
        (server-side restart, network blip, provider rate-limit kick)
        automatically reconnect with exponential backoff. The
        TTL-gated REST fallback in the realtime task closes the data
        gap during a reconnect.

        Falls back gracefully if the provider does not support
        WebSocket (returns without doing anything).
        """
        if not hasattr(self._provider, "supports_websocket") or not self._provider.supports_websocket():
            logger.info(
                "Provider %s does not support WebSocket; buffer will use REST only.",
                self._provider.name,
            )
            return

        self._streaming = True
        tasks: list[Any] = []
        for symbol in self._symbols:
            # Capture loop-local closures so each reconnect spins up a
            # fresh subscribe call rather than re-awaiting an exhausted
            # coroutine. Bind symbol/interval at def time, not call time.
            def _make_bars_factory(sym: str, interval: str) -> Any:
                async def _factory() -> None:
                    await self._provider.subscribe_bars(
                        sym, interval, self._on_bar
                    )
                return _factory

            tasks.append(
                self._subscribe_with_reconnect(
                    f"bars/{symbol}",
                    _make_bars_factory(symbol, self._interval),
                )
            )
            if hasattr(self._provider, "subscribe_order_book"):
                def _make_ob_factory(sym: str) -> Any:
                    async def _factory() -> None:
                        await self._provider.subscribe_order_book(
                            sym, self._on_order_book
                        )
                    return _factory

                tasks.append(
                    self._subscribe_with_reconnect(
                        f"ob/{symbol}",
                        _make_ob_factory(symbol),
                    )
                )
        await asyncio.gather(*tasks, return_exceptions=False)

    def start_streaming_background(self) -> threading.Thread | None:
        """Launch :meth:`start_streaming` in a daemon thread.

        Returns the thread (or ``None`` if the provider lacks WebSocket).
        """
        if not hasattr(self._provider, "supports_websocket") or not self._provider.supports_websocket():
            return None

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.start_streaming())
            except Exception:
                logger.warning("WebSocket streaming stopped", exc_info=True)
            finally:
                loop.close()

        t = threading.Thread(target=_run, daemon=True, name="mdb-ws-stream")
        t.start()
        return t

    def wait_for_warmup(self, timeout: float = 5.0) -> bool:
        """Block until every tracked symbol has received its first
        WebSocket push (bar or order-book), or ``timeout`` seconds
        elapse total.

        Returns ``True`` if every symbol warmed up in time, ``False``
        otherwise. A False return is informational only — callers
        usually log a warning and continue, relying on REST fallbacks
        until the WS catches up. Symbols that warmed up before the
        deadline keep their event set; the helper short-circuits
        immediately on subsequent calls.

        Implementation: iterates symbol-by-symbol with a per-symbol
        deadline derived from the shared budget. This is the simplest
        correct shape; we don't need true any-of semantics because the
        callsite (task ``load_data``) wants ALL symbols ready before
        the gym loop starts.
        """
        if not self._warmup_events:
            return True
        deadline = time.monotonic() + max(0.0, float(timeout))
        all_warm = True
        for symbol, event in self._warmup_events.items():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                if not event.is_set():
                    all_warm = False
                continue
            if not event.wait(remaining):
                all_warm = False
        return all_warm

    # Read API

    def get_recent_bars(self, symbol: str, n: int) -> list[MarketSnapshot]:
        """Return the last *n* bars for *symbol*, sorted chronologically."""
        with self._lock:
            buf = self._bars.get(symbol, OrderedDict())
            items = list(buf.values())
        return items[-n:]

    def get_latest_price(self, symbol: str) -> MarketSnapshot | None:
        """Return the most recent snapshot for *symbol*, or ``None``."""
        with self._lock:
            buf = self._bars.get(symbol, OrderedDict())
            if not buf:
                return None
            # Last item is the most recent (OrderedDict preserves order)
            return next(reversed(buf.values()))

    def price_staleness_ms(self, symbol: str) -> float:
        """Return elapsed receive age, or infinity before the first bar.

        Uses perf_counter for elapsed-time measurement.
        """
        with self._lock:
            recv = self._bar_recv_perf.get(symbol, 0.0)
        if recv == 0.0:
            return float("inf")
        return (time.perf_counter() - recv) * 1000.0

    def ob_staleness_ms(self, symbol: str) -> float:
        """Milliseconds since this symbol's last order-book snapshot
        arrived. ``inf`` until the first push lands."""
        with self._lock:
            recv = self._ob_recv_perf.get(symbol, 0.0)
        if recv == 0.0:
            return float("inf")
        return (time.perf_counter() - recv) * 1000.0

    def has_order_book(self, symbol: str) -> bool:
        """Return True if a WS-pushed (or otherwise-cached) order book
        exists for *symbol*.

        Useful for callers that want to know whether to pre-fetch the
        order book via REST in parallel with other calls vs. wait for
        WS to push one. The check itself is cheap and takes the read
        lock just long enough to peek at the dict.
        """
        with self._lock:
            return self._order_books.get(symbol) is not None

    def get_order_book(self, symbol: str) -> OrderBookSnapshot | None:
        """Return the latest order-book snapshot.

        Prefers WebSocket-pushed data; falls back to a REST call if the
        provider exposes ``get_order_book``.
        """
        with self._lock:
            ob = self._order_books.get(symbol)
        if ob is not None:
            return ob
        if hasattr(self._provider, "get_order_book"):
            try:
                return self._provider.get_order_book(symbol)
            except Exception:
                logger.debug("REST order-book fetch failed for %s", symbol, exc_info=True)
        return None
