#!/usr/bin/env python3
"""Benchmark realtime trading operations and end-to-end task latency.

The ``local`` profile is synthetic. ``binance`` and ``alpaca`` add live market
data probes; optional Alpaca paper-order mutations require explicit opt-in.
This is a measurement tool, not a correctness gate.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_dotenv_best_effort() -> None:
    """Load API keys from the working directory or repository ``.env``."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()  # cwd
    repo_env = _REPO_ROOT / ".env"
    if repo_env.exists():
        load_dotenv(dotenv_path=repo_env, override=False)


_load_dotenv_best_effort()


from openfinai_pipeline.realtime.data_providers.base import (  # noqa: E402
    MarketSnapshot,
    OrderBookSnapshot,
    downsample_bars,
)
from openfinai_pipeline.realtime.execution import (  # noqa: E402
    ActionVerb,
    OrderIntent,
    OrderType,
    SimulatedExecutor,
    TimeInForce,
)
from openfinai_pipeline.realtime.market_data_buffer import MarketDataBuffer  # noqa: E402
from openfinai_pipeline.realtime.tasks.realtime_trading_task import (  # noqa: E402
    RealtimeTradingTask,
)
from openfinai_pipeline.realtime.agent_runtime import _to_jsonable  # noqa: E402
from openfinai_pipeline.benchmark.contracts import (  # noqa: E402
    _compute_trading_metrics_from_history,
)
from openfinai_pipeline.settings import TradingConfig  # noqa: E402

from data.knowledge_base.rewards import (  # noqa: E402
    MaxDrawdown,
    PnL,
    SharpeRatio,
    WinRate,
)


@dataclass
class Sample:
    category: str
    name: str
    n_iterations: int
    timings_ns: list[int] = field(default_factory=list)
    notes: str = ""
    error: str | None = None

    def percentiles_us(self) -> dict[str, float]:
        if self.error is not None or not self.timings_ns:
            return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"),
                    "p99": float("nan"), "max": float("nan"), "min": float("nan")}
        ns = sorted(self.timings_ns)
        n = len(ns)

        def _pct(p: float) -> float:
            # Linear-interpolation percentile, mirrors numpy default.
            if n == 1:
                return ns[0] / 1000.0
            k = (n - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return ns[int(k)] / 1000.0
            return (ns[f] + (ns[c] - ns[f]) * (k - f)) / 1000.0

        return {
            "mean": statistics.fmean(ns) / 1000.0,
            "p50": _pct(0.50),
            "p95": _pct(0.95),
            "p99": _pct(0.99),
            "max": ns[-1] / 1000.0,
            "min": ns[0] / 1000.0,
        }


def measure(
    category: str,
    name: str,
    fn: Callable[[], Any],
    *,
    iterations: int,
    warmup: int = 5,
    notes: str = "",
) -> Sample:
    """Measure ``fn`` after warmup, with GC disabled during sampling."""
    sample = Sample(category=category, name=name, n_iterations=iterations, notes=notes)
    try:
        for _ in range(max(warmup, 0)):
            fn()
    except Exception as exc:
        sample.error = f"warmup raised: {type(exc).__name__}: {exc}"
        return sample

    gc.collect()
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            try:
                fn()
            except Exception as exc:
                sample.error = f"call raised: {type(exc).__name__}: {exc}"
                break
            t1 = time.perf_counter_ns()
            sample.timings_ns.append(t1 - t0)
    finally:
        if gc_was_enabled:
            gc.enable()
    return sample


class SyntheticProvider:
    """Deterministic in-process market-data provider."""

    name = "synthetic"

    def __init__(self, price: float = 100.0) -> None:
        self._price = price
        self.call_count_get_current_price = 0
        self.call_count_get_bars = 0
        self.call_count_get_order_book = 0

    def _snap(self, symbol: str) -> MarketSnapshot:
        now = datetime.now(timezone.utc)
        return MarketSnapshot(
            symbol=symbol,
            timestamp=now.replace(second=0, microsecond=0),
            price=self._price,
            open=self._price,
            high=self._price + 0.5,
            low=self._price - 0.5,
            close=self._price,
            volume=1234.5,
        )

    def get_current_price(self, symbol: str) -> MarketSnapshot:
        self.call_count_get_current_price += 1
        return self._snap(symbol)

    def get_price_at(self, symbol: str, at: datetime) -> MarketSnapshot:
        return MarketSnapshot(symbol=symbol, timestamp=at, price=self._price)

    def get_bars(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[MarketSnapshot]:
        self.call_count_get_bars += 1
        # Bound synthetic history to keep benchmark time predictable.
        bars: list[MarketSnapshot] = []
        cursor = start.replace(second=0, microsecond=0)
        i = 0
        max_bars = 600
        while cursor <= end and i < max_bars:
            bars.append(
                MarketSnapshot(
                    symbol=symbol,
                    timestamp=cursor,
                    price=self._price + (i % 11) * 0.1,
                    open=self._price,
                    high=self._price + 1.0,
                    low=self._price - 1.0,
                    close=self._price + (i % 11) * 0.1,
                    volume=1000.0 + i,
                )
            )
            cursor += timedelta(minutes=1)
            i += 1
        return bars

    def get_order_book(self, symbol: str, depth: int = 20) -> OrderBookSnapshot:
        self.call_count_get_order_book += 1
        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            best_bid=self._price - 0.05,
            best_bid_qty=12.0,
            best_ask=self._price + 0.05,
            best_ask_qty=14.0,
            bids=[(self._price - 0.05 - i * 0.01, 5.0) for i in range(depth)],
            asks=[(self._price + 0.05 + i * 0.01, 5.0) for i in range(depth)],
        )

    def supports_websocket(self) -> bool:  # noqa: PLR6301
        return False


def _seed_buffer(buf: MarketDataBuffer, symbol: str, n_bars: int, base_price: float) -> None:
    """Populate *buf* with ``n_bars`` synthetic bars (chronological)."""
    t0 = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    for i in range(n_bars):
        bar = MarketSnapshot(
            symbol=symbol,
            timestamp=t0 - timedelta(minutes=n_bars - i),
            price=base_price + (i % 17) * 0.1,
            open=base_price,
            high=base_price + 1.0,
            low=base_price - 1.0,
            close=base_price + (i % 17) * 0.1,
            volume=1000.0 + i,
        )
        buf.insert_bar(symbol, bar)


def _bench_in_memory_suite(symbol: str, iterations: int, warmup: int) -> list[Sample]:
    """Run every pure-CPU microbenchmark. No network."""
    samples: list[Sample] = []

    intent_dicts = {
        "market_buy": {"action": "buy", "symbol": symbol, "quantity": 1.0},
        "market_sell": {"action": "sell", "symbol": symbol, "quantity": 1.0},
        "hold": {"action": "hold", "symbol": symbol, "quantity": 0.0},
        "limit_gtc": {
            "action": "buy", "symbol": symbol, "quantity": 1.0,
            "order_type": "limit", "limit_price": 100.0, "tif": "gtc",
        },
        "stop_limit_gtc": {
            "action": "sell", "symbol": symbol, "quantity": 1.0,
            "order_type": "stop_limit", "stop_price": 99.0, "limit_price": 98.5,
            "tif": "gtc",
        },
        "malformed_missing_action": {"symbol": symbol, "quantity": 1.0},
    }
    for label, d in intent_dicts.items():
        captured = d  # avoid late-binding pitfall
        samples.append(measure(
            "OrderIntent",
            f"from_dict[{label}]",
            lambda d=captured: OrderIntent.from_dict(d),
            iterations=iterations, warmup=warmup,
        ))

    cfg = TradingConfig(slippage_pct=0.001, transaction_cost_pct=0.0,
                        execution_mode="internal_paper")
    market_price = 100.0

    def _fresh_executor(seed_fills: int = 0, seed_pending: int = 0) -> SimulatedExecutor:
        ex = SimulatedExecutor(cfg, initial_capital=1_000_000.0)
        # Seed non-empty histories and pending queues when requested.
        for i in range(seed_fills):
            intent = OrderIntent(
                action=ActionVerb.BUY, symbol=symbol, quantity=1.0,
                order_type=OrderType.MARKET, tif=TimeInForce.IOC,
            )
            ex.submit(intent, market_price=market_price + (i % 7) * 0.1, step=i)
        for j in range(seed_pending):
            far_limit = market_price * 0.5  # never marketable
            intent = OrderIntent(
                action=ActionVerb.BUY, symbol=symbol, quantity=1.0,
                order_type=OrderType.LIMIT, limit_price=far_limit, tif=TimeInForce.GTC,
            )
            ex.submit(intent, market_price=market_price, step=j)
        return ex

    market_buy_intent = OrderIntent(
        action=ActionVerb.BUY, symbol=symbol, quantity=1.0,
        order_type=OrderType.MARKET, tif=TimeInForce.IOC,
    )
    hold_intent = OrderIntent(
        action=ActionVerb.HOLD, symbol=symbol, quantity=0.0,
        order_type=OrderType.MARKET, tif=TimeInForce.IOC,
    )
    far_limit_intent_factory = lambda: OrderIntent(  # noqa: E731
        action=ActionVerb.BUY, symbol=symbol, quantity=1.0,
        order_type=OrderType.LIMIT, limit_price=50.0, tif=TimeInForce.GTC,
    )

    def _submit_market_buy_iter() -> None:
        ex = _fresh_executor()
        ex.submit(market_buy_intent, market_price=market_price)

    samples.append(measure(
        "SimulatedExecutor", "submit[market_buy, fresh]",
        _submit_market_buy_iter, iterations=iterations, warmup=warmup,
        notes="includes ~1 dict alloc + cash debit + position upsert",
    ))

    hot_exec = _fresh_executor(seed_fills=100)
    samples.append(measure(
        "SimulatedExecutor", "submit[market_buy, hot100]",
        lambda: hot_exec.submit(market_buy_intent, market_price=market_price),
        iterations=iterations, warmup=warmup,
        notes="executor primed with 100 prior fills",
    ))

    samples.append(measure(
        "SimulatedExecutor", "submit[hold]",
        lambda: _fresh_executor().submit(hold_intent, market_price=market_price),
        iterations=iterations, warmup=warmup,
    ))

    def _submit_queue_limit() -> None:
        ex = _fresh_executor()
        ex.submit(far_limit_intent_factory(), market_price=market_price)

    samples.append(measure(
        "SimulatedExecutor", "submit[limit, queued]",
        _submit_queue_limit, iterations=iterations, warmup=warmup,
    ))

    for n_pending in (0, 10, 100):
        ex = _fresh_executor(seed_pending=n_pending)
        prices = {symbol: market_price}
        samples.append(measure(
            "SimulatedExecutor", f"tick[pending={n_pending}]",
            lambda ex=ex, prices=prices: ex.tick(prices),
            iterations=iterations, warmup=warmup,
        ))

    def _cancel_iter() -> None:
        ex = _fresh_executor(seed_pending=1)
        pending = ex.get_pending_orders()
        ex.cancel(pending[0].order_id)
    samples.append(measure(
        "SimulatedExecutor", "cancel[1 pending]",
        _cancel_iter, iterations=iterations, warmup=warmup,
        notes="includes the fresh-executor setup tax",
    ))

    for n_fills in (0, 10, 100):
        ex = _fresh_executor(seed_fills=n_fills)
        prices = {symbol: market_price + 0.5}
        samples.append(measure(
            "SimulatedExecutor", f"compute_pnl[fills={n_fills}]",
            lambda ex=ex, prices=prices: ex.compute_pnl(prices),
            iterations=iterations, warmup=warmup,
        ))

    ex_steady = _fresh_executor(seed_fills=100, seed_pending=20)
    samples.append(measure(
        "SimulatedExecutor", "get_positions",
        ex_steady.get_positions, iterations=iterations, warmup=warmup,
    ))
    samples.append(measure(
        "SimulatedExecutor", "get_pending_orders[20]",
        ex_steady.get_pending_orders, iterations=iterations, warmup=warmup,
    ))
    samples.append(measure(
        "SimulatedExecutor", "get_cash",
        ex_steady.get_cash, iterations=iterations, warmup=warmup,
    ))
    samples.append(measure(
        "SimulatedExecutor", "get_reserved_cash",
        ex_steady.get_reserved_cash, iterations=iterations, warmup=warmup,
    ))

    provider = SyntheticProvider()
    buf = MarketDataBuffer(provider=provider, symbols=[symbol], interval="1m",
                           buffer_size=500, backfill_bars=200)
    _seed_buffer(buf, symbol, n_bars=200, base_price=market_price)

    new_bar_factory = lambda: MarketSnapshot(  # noqa: E731
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),  # fresh ts → new buffer entry
        price=market_price,
        volume=1.0,
    )
    samples.append(measure(
        "MarketDataBuffer", "insert_bar[primed]",
        lambda: buf.insert_bar(symbol, new_bar_factory()),
        iterations=iterations, warmup=warmup,
        notes="includes lock acquire + dedup + LRU eviction",
    ))

    samples.append(measure(
        "MarketDataBuffer", "get_latest_price[primed]",
        lambda: buf.get_latest_price(symbol),
        iterations=iterations, warmup=warmup,
    ))

    for n in (10, 60, 200):
        samples.append(measure(
            "MarketDataBuffer", f"get_recent_bars[n={n}]",
            lambda n=n: buf.get_recent_bars(symbol, n),
            iterations=iterations, warmup=warmup,
        ))

    samples.append(measure(
        "MarketDataBuffer", "get_order_book[REST fallback]",
        lambda: buf.get_order_book(symbol),
        iterations=iterations, warmup=warmup,
        notes="synthetic provider returns immediately; real provider hits REST",
    ))

    primary_bars = buf.get_recent_bars(symbol, 200)
    samples.append(measure(
        "downsample_bars", "1m→5m, last 20",
        lambda: downsample_bars(primary_bars, "5m", 20),
        iterations=iterations, warmup=warmup,
    ))
    samples.append(measure(
        "downsample_bars", "1m→15m, last 40",
        lambda: downsample_bars(primary_bars, "15m", 40),
        iterations=iterations, warmup=warmup,
    ))
    samples.append(measure(
        "downsample_bars", "1m→1h, last 24",
        lambda: downsample_bars(primary_bars, "1h", 24),
        iterations=iterations, warmup=warmup,
    ))

    fake_history: list[dict[str, Any]] = []
    for i in range(200):
        fake_history.append({
            "step": i,
            "action_type": "buy" if i % 3 == 0 else ("sell" if i % 3 == 1 else "hold"),
            "symbol": symbol,
            "quantity": 1.0,
            "market_price": market_price + (i % 7) * 0.1,
            "executed_price": market_price + (i % 7) * 0.1,
            "reward": ((i % 5) - 2) * 0.1,
        })
    reward_classes = (PnL, SharpeRatio, MaxDrawdown, WinRate)
    samples.append(measure(
        "reward_bank", "_compute_trading_metrics_from_history[hist=200]",
        lambda: _compute_trading_metrics_from_history(
            fake_history, reward_classes, count_label="num_steps"
        ),
        iterations=iterations, warmup=warmup,
        notes="end-of-trial; runs once per episode",
    ))

    rich_obs: dict[str, Any] = {
        "symbols": {
            symbol: {
                "symbol": symbol,
                "price": market_price,
                "volume": 1234.5,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recent_bars": buf.get_recent_bars(symbol, 60),
                "order_book": provider.get_order_book(symbol),
                "recent_bars_by_interval": {
                    "1m": buf.get_recent_bars(symbol, 60),
                    "5m": downsample_bars(primary_bars, "5m", 20),
                },
            }
        },
        "positions": {symbol: 1.0},
        "pnl": {symbol: 0.5, "__total": 0.5},
        "cash": 999_000.0,
        "reserved_cash": 0.0,
        "pending_orders": [],
    }
    samples.append(measure(
        "agent_runtime", "_to_jsonable[full obs, 60 bars + 5m sidecar]",
        lambda: _to_jsonable(rich_obs),
        iterations=iterations, warmup=warmup,
        notes="per step; serialises observation handed to train.step()",
    ))

    task = RealtimeTradingTask(
        provider=provider,
        symbols=[symbol],
        trading_config=cfg,
        context_resolutions=[
            {"interval": "1m", "bars": 60},
            {"interval": "5m", "bars": 12},
        ],
        data_resolution="1m",
        buffer_size=500,
        max_steps=0,  # unlimited so done flag doesn't trip the benchmark
        initial_capital=1_000_000.0,
    )
    task.reset()

    def _step_iter() -> None:
        i = _step_iter._counter  # type: ignore[attr-defined]
        if i % 4 == 0:
            action = {"action": "buy", "symbol": symbol, "quantity": 0.01}
        elif i % 4 == 1:
            action = {"action": "hold", "symbol": symbol, "quantity": 0.0}
        elif i % 4 == 2:
            action = {"action": "sell", "symbol": symbol, "quantity": 0.005}
        else:
            action = {"action": "hold", "symbol": symbol, "quantity": 0.0}
        _step_iter._counter = i + 1  # type: ignore[attr-defined]
        task.step(action)
    _step_iter._counter = 0  # type: ignore[attr-defined]

    samples.append(measure(
        "RealtimeTradingTask", "step[end-to-end, synthetic provider]",
        _step_iter, iterations=iterations, warmup=warmup,
        notes=(
            "AGENT-ACTION LATENCY (framework side): one step() = obs build + "
            "executor tick + submit + PnL recompute. Excludes the agent's own "
            "decision compute, which is unbounded and lives in train.py.step()"
        ),
    ))

    samples.append(measure(
        "RealtimeTradingTask", "_get_market_observation[1 symbol, 1m+5m]",
        task._get_market_observation, iterations=iterations, warmup=warmup,
        notes="excludes executor.tick + .submit",
    ))

    return samples


def _bench_integrated_step(
    category_label: str,
    provider: Any,
    symbol: str,
    iterations: int,
    warmup: int,
) -> list[Sample]:
    """Measure a task step with live data and in-memory paper execution."""
    samples: list[Sample] = []
    cfg = TradingConfig(slippage_pct=0.001, transaction_cost_pct=0.0,
                        execution_mode="internal_paper")

    task = RealtimeTradingTask(
        provider=provider,
        symbols=[symbol],
        trading_config=cfg,
        context_resolutions=[{"interval": "1m", "bars": 60}],
        data_resolution="1m",
        buffer_size=500,
        max_steps=0,
        initial_capital=1_000_000.0,
    )
    # Disable asynchronous updates so they cannot contaminate timed steps.
    task._buffer.start_streaming_background = lambda: None  # type: ignore[method-assign]

    reset_iters = max(1, min(iterations // 5, 3))

    def _reset_iter() -> None:
        task._data = None
        task.reset()
    samples.append(measure(
        category_label, "task.reset[with backfill]",
        _reset_iter, iterations=reset_iters, warmup=0,
        notes=f"per-trial setup cost; ran {reset_iters} times",
    ))

    if task._data is None:
        task.reset()

    counter = {"i": 0}

    def _step_iter() -> None:
        i = counter["i"]
        if i % 3 == 0:
            action = {"action": "buy", "symbol": symbol, "quantity": 0.001}
        elif i % 3 == 1:
            action = {"action": "sell", "symbol": symbol, "quantity": 0.0005}
        else:
            action = {"action": "hold", "symbol": symbol, "quantity": 0.0}
        counter["i"] = i + 1
        task.step(action)

    samples.append(measure(
        category_label, "task.step[real provider, internal_paper]",
        _step_iter, iterations=iterations, warmup=min(warmup, 2),
        notes=(
            "AGENT-ACTION LATENCY (framework side, internal_paper): one step() = "
            "obs build + executor.tick/submit (in-memory) + PnL recompute. Price "
            "is read from the WS-fed buffer (REST get_current_price fires only when "
            "buffer staleness > TTL). Excludes the agent's own decision compute"
        ),
    ))
    return samples


def _measure_binance_clock_offset_ms(provider: Any) -> float:
    """Return the median Binance-server-minus-local clock offset."""
    offsets: list[float] = []
    for _ in range(5):
        t0 = time.time() * 1000.0
        resp = provider._get("/api/v3/time")
        t1 = time.time() * 1000.0
        offsets.append(float(resp["serverTime"]) - (t0 + t1) / 2.0)
        time.sleep(0.05)
    return statistics.median(offsets)


def _bench_binance_ws(symbol: str, provider: Any, ws_seconds: float) -> list[Sample]:
    """Measure Binance aggregate-trade and book-ticker WebSocket cadence."""
    import asyncio
    import json as _json

    try:
        offset_ms = _measure_binance_clock_offset_ms(provider)
    except Exception:
        offset_ms = 0.0

    async def _record(stream: str, want_event_time: bool):
        import websockets

        url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@{stream}"
        one_way_ns: list[int] = []
        inter_ns: list[int] = []
        last_recv_perf: int | None = None
        deadline = time.monotonic() + ws_seconds
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                while time.monotonic() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    t_recv_perf_ns = time.perf_counter_ns()
                    t_recv_ms_aligned = time.time() * 1000.0 + offset_ms
                    if last_recv_perf is not None:
                        inter_ns.append(t_recv_perf_ns - last_recv_perf)
                    last_recv_perf = t_recv_perf_ns
                    if not want_event_time:
                        continue
                    try:
                        msg = _json.loads(raw)
                        E = msg.get("E")
                    except Exception:
                        continue
                    if E is None:
                        continue
                    one_way_ns.append(
                        int((t_recv_ms_aligned - float(E)) * 1_000_000.0)
                    )
        except Exception as exc:
            return one_way_ns, inter_ns, repr(exc)
        return one_way_ns, inter_ns, None

    async def _run():
        return await asyncio.gather(
            _record("aggTrade", want_event_time=True),
            _record("bookTicker", want_event_time=False),
        )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        (agg_one, agg_inter, agg_err), (book_one, book_inter, book_err) = (
            loop.run_until_complete(_run())
        )
    finally:
        loop.close()

    samples: list[Sample] = []
    note_oneway = (
        f"window={ws_seconds:.0f}s; clock offset (binance−local) "
        f"= {offset_ms:+.0f}ms"
    )
    note_inter = f"window={ws_seconds:.0f}s; cadence between consecutive WS msgs"

    if agg_one:
        samples.append(Sample(
            category="BinanceProvider WS @aggTrade",
            name="one-way (E→local recv)",
            n_iterations=len(agg_one),
            timings_ns=agg_one, notes=note_oneway,
        ))
    if agg_inter:
        samples.append(Sample(
            category="BinanceProvider WS @aggTrade",
            name="inter-arrival",
            n_iterations=len(agg_inter),
            timings_ns=agg_inter, notes=note_inter,
        ))
    if book_inter:
        samples.append(Sample(
            category="BinanceProvider WS @bookTicker",
            name="inter-arrival",
            n_iterations=len(book_inter),
            timings_ns=book_inter,
            notes=(
                f"{note_inter}; no E field on @bookTicker so one-way "
                "latency is unmeasurable from-message"
            ),
        ))
    if agg_err is not None or book_err is not None:
        samples.append(Sample(
            category="BinanceProvider WS", name="(errors)", n_iterations=0,
            error=f"aggTrade={agg_err!r} bookTicker={book_err!r}",
        ))
    return samples


def _bench_binance(symbol: str, iterations: int, warmup: int,
                   *, ws_seconds: float = 15.0) -> list[Sample]:
    from openfinai_pipeline.realtime.data_providers.binance import BinanceProvider
    provider = BinanceProvider()
    samples: list[Sample] = []
    samples.append(measure(
        "BinanceProvider", "get_current_price",
        lambda: provider.get_current_price(symbol),
        iterations=iterations, warmup=warmup,
        notes="public REST, no auth; rate-limit throttle inline",
    ))
    samples.append(measure(
        "BinanceProvider", "get_order_book[depth=20]",
        lambda: provider.get_order_book(symbol, depth=20),
        iterations=iterations, warmup=warmup,
    ))

    backfill_iters = max(1, min(iterations // 5, 5))
    start = datetime.now(timezone.utc) - timedelta(minutes=200)
    end = datetime.now(timezone.utc)
    samples.append(measure(
        "BinanceProvider", "get_bars[1m, 200 bars]",
        lambda: provider.get_bars(symbol, "1m", start, end),
        iterations=backfill_iters, warmup=min(warmup, 1),
        notes=f"per-trial backfill cost; ran {backfill_iters} times",
    ))

    if ws_seconds > 0:
        samples.extend(_bench_binance_ws(symbol, provider, ws_seconds))

    samples.extend(_bench_integrated_step(
        "RealtimeTradingTask[binance]", provider, symbol, iterations, warmup,
    ))
    return samples


def _bench_alpaca_paper_executor(
    iterations: int,
    warmup: int,
    *,
    include_mutations: bool,
) -> list[Sample]:
    """Measure Alpaca paper REST calls.

    Mutations are opt-in and create visible paper orders before cancelling them.
    ``reset()`` is excluded because it closes all paper-account positions.
    """
    from openfinai_pipeline.realtime.execution.alpaca_paper import AlpacaPaperExecutor

    samples: list[Sample] = []
    try:
        executor = AlpacaPaperExecutor()
    except Exception as exc:
        return [Sample(
            category="AlpacaPaperExecutor",
            name="(setup failed)",
            n_iterations=0,
            error=f"{type(exc).__name__}: {exc}",
        )]

    samples.append(measure(
        "AlpacaPaperExecutor", "get_cash",
        executor.get_cash, iterations=iterations, warmup=warmup,
        notes="GET /v2/account",
    ))
    samples.append(measure(
        "AlpacaPaperExecutor", "get_positions",
        executor.get_positions, iterations=iterations, warmup=warmup,
        notes="GET /v2/positions",
    ))
    samples.append(measure(
        "AlpacaPaperExecutor", "get_pending_orders",
        executor.get_pending_orders, iterations=iterations, warmup=warmup,
        notes="GET /v2/orders?status=open",
    ))
    samples.append(measure(
        "AlpacaPaperExecutor", "compute_pnl",
        lambda: executor.compute_pnl({}),
        iterations=iterations, warmup=warmup,
        notes="GET /v2/positions; current_prices arg is read-only fallback",
    ))
    samples.append(measure(
        "AlpacaPaperExecutor", "tick",
        lambda: executor.tick({}),
        iterations=iterations, warmup=warmup,
        notes="GET /v2/orders?status=all; polls for newly-resolved orders",
    ))

    if include_mutations:
        # Use a valid, deliberately remote limit price and cancel promptly.
        mut_symbol = "SPY"
        mut_intent = OrderIntent(
            action=ActionVerb.BUY, symbol=mut_symbol, quantity=1.0,
            order_type=OrderType.LIMIT, limit_price=1.00, tif=TimeInForce.GTC,
        )

        # Limit mutation volume and pair every accepted order with a cancel.
        mut_iters = min(iterations, 10)
        submit_ids: list[str] = []

        def _submit_iter() -> None:
            result = executor.submit(mut_intent, market_price=400.0)
            if result.is_accepted and result.accepted is not None:
                submit_ids.append(result.accepted.order_id)

        samples.append(measure(
            "AlpacaPaperExecutor", "submit[limit, far-from-market]",
            _submit_iter, iterations=mut_iters, warmup=0,
            notes=(
                f"POST /v2/orders (real paper order!); ran {mut_iters} times. "
                "Each is paired with a cancel below."
            ),
        ))

        if submit_ids:
            cancel_iter = iter(submit_ids)

            def _cancel_iter() -> None:
                oid = next(cancel_iter, None)
                if oid is None:
                    return
                executor.cancel(oid)

            samples.append(measure(
                "AlpacaPaperExecutor", "cancel",
                _cancel_iter, iterations=len(submit_ids), warmup=0,
                notes=f"DELETE /v2/orders/<id>; cancels the {len(submit_ids)} submits above",
            ))

        # Remove any accepted order left behind by a failed cancellation.
        samples.append(measure(
            "AlpacaPaperExecutor", "expire_all",
            lambda: executor.expire_all(),
            iterations=1, warmup=0,
            notes="cleanup: cancel any leftover local-pending orders",
        ))

    return samples


def _bench_alpaca_ws(
    symbol: str, ws_seconds: float, api_key: str, api_secret: str,
    *, binance_offset_ms: float = 0.0,
) -> list[Sample]:
    """Measure Alpaca IEX trade and quote WebSocket streams.

    One-way results depend on local clock accuracy. A Binance offset is used when
    available; both channels share one Alpaca connection.
    """
    import asyncio
    import json as _json

    api_key = (api_key or "").split("#", 1)[0].strip()
    api_secret = (api_secret or "").split("#", 1)[0].strip()

    trades_one: list[int] = []
    trades_inter: list[int] = []
    quotes_one: list[int] = []
    quotes_inter: list[int] = []
    last_trade_perf: int | None = None
    last_quote_perf: int | None = None
    error: str | None = None

    async def _consume():
        import websockets
        nonlocal last_trade_perf, last_quote_perf
        url = "wss://stream.data.alpaca.markets/v2/iex"
        async with websockets.connect(url, ping_interval=20) as ws:
            await ws.send(_json.dumps({
                "action": "auth", "key": api_key, "secret": api_secret,
            }))
            await asyncio.wait_for(ws.recv(), timeout=10.0)
            await ws.send(_json.dumps({
                "action": "subscribe",
                "trades": [symbol], "quotes": [symbol],
            }))
            await asyncio.wait_for(ws.recv(), timeout=5.0)

            deadline = time.monotonic() + ws_seconds
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                t_recv_perf = time.perf_counter_ns()
                t_recv_ms = time.time() * 1000.0 + binance_offset_ms
                payload = _json.loads(raw)
                if not isinstance(payload, list):
                    payload = [payload]
                for item in payload:
                    kind = item.get("T")
                    ts_str = item.get("t")
                    if kind not in ("t", "q") or ts_str is None:
                        continue
                    try:
                        ts = datetime.fromisoformat(
                            str(ts_str).replace("Z", "+00:00")
                        )
                    except Exception:
                        continue
                    event_ms = ts.timestamp() * 1000.0
                    one_way_ns = int(
                        (t_recv_ms - event_ms) * 1_000_000.0
                    )
                    if kind == "t":
                        trades_one.append(one_way_ns)
                        if last_trade_perf is not None:
                            trades_inter.append(
                                t_recv_perf - last_trade_perf
                            )
                        last_trade_perf = t_recv_perf
                    else:
                        quotes_one.append(one_way_ns)
                        if last_quote_perf is not None:
                            quotes_inter.append(
                                t_recv_perf - last_quote_perf
                            )
                        last_quote_perf = t_recv_perf

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_consume())
    except Exception as exc:
        error = repr(exc)
    finally:
        loop.close()

    samples: list[Sample] = []
    note_oneway = (
        f"window={ws_seconds:.0f}s; clock-aligned to local "
        f"(+binance offset {binance_offset_ms:+.0f}ms)"
    )
    note_inter = (
        f"window={ws_seconds:.0f}s; cadence between consecutive WS msgs"
    )
    for label, oneway, inter in (
        ("trades", trades_one, trades_inter),
        ("quotes", quotes_one, quotes_inter),
    ):
        if oneway:
            samples.append(Sample(
                category=f"AlpacaProvider WS {label}",
                name="one-way (server t→local recv)",
                n_iterations=len(oneway),
                timings_ns=oneway, notes=note_oneway,
            ))
        if inter:
            samples.append(Sample(
                category=f"AlpacaProvider WS {label}",
                name="inter-arrival",
                n_iterations=len(inter),
                timings_ns=inter, notes=note_inter,
            ))
    # Preserve a diagnostic row when the stream is silent.
    if not (trades_one or quotes_one):
        diag = error if error else (
            f"no trades or quotes received from IEX in {ws_seconds:.0f}s "
            "window — symbol may be in a quiet period or WS subscribed "
            "to wrong channels"
        )
        samples.append(Sample(
            category="AlpacaProvider WS", name="(no data captured)",
            n_iterations=0, error=diag,
        ))
    return samples


def _bench_alpaca(
    symbol: str,
    iterations: int,
    warmup: int,
    *,
    include_paper_executor: bool = True,
    include_paper_mutations: bool = False,
    ws_seconds: float = 15.0,
    binance_offset_ms: float = 0.0,
) -> list[Sample]:
    if not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")):
        return [Sample(
            category="AlpacaProvider",
            name="(skipped — set ALPACA_API_KEY + ALPACA_SECRET_KEY)",
            n_iterations=0,
            error="missing credentials",
        )]
    from openfinai_pipeline.realtime.data_providers.alpaca import AlpacaProvider
    provider = AlpacaProvider()
    samples: list[Sample] = []
    samples.append(measure(
        "AlpacaProvider", "get_current_price",
        lambda: provider.get_current_price(symbol),
        iterations=iterations, warmup=warmup,
        notes="IEX feed, free-tier auth",
    ))
    samples.append(measure(
        "AlpacaProvider", "get_order_book[NBBO]",
        lambda: provider.get_order_book(symbol),
        iterations=iterations, warmup=warmup,
        notes="Alpaca exposes NBBO only (no L2 depth)",
    ))
    start = datetime.now(timezone.utc) - timedelta(minutes=200)
    end = datetime.now(timezone.utc)
    backfill_iters = max(1, min(iterations // 5, 5))
    samples.append(measure(
        "AlpacaProvider", "get_bars[1Min, 200 bars]",
        lambda: provider.get_bars(symbol, "1m", start, end),
        iterations=backfill_iters, warmup=min(warmup, 1),
        notes=f"per-trial backfill cost; ran {backfill_iters} times",
    ))

    if ws_seconds > 0:
        samples.extend(_bench_alpaca_ws(
            symbol, ws_seconds,
            api_key=os.environ.get("ALPACA_API_KEY", ""),
            api_secret=os.environ.get("ALPACA_SECRET_KEY", ""),
            binance_offset_ms=binance_offset_ms,
        ))

    samples.extend(_bench_integrated_step(
        "RealtimeTradingTask[alpaca]", provider, symbol, iterations, warmup,
    ))

    if include_paper_executor:
        samples.extend(_bench_alpaca_paper_executor(
            iterations, warmup, include_mutations=include_paper_mutations,
        ))
    return samples


def _fmt_us(us: float) -> str:
    if math.isnan(us):
        return "    nan"
    if us < 1.0:
        return f"{us * 1000:7.0f}ns"
    if us < 1000.0:
        return f"{us:7.2f}µs"
    if us < 1_000_000.0:
        return f"{us / 1000:7.2f}ms"
    return f"{us / 1_000_000:7.2f}s "


def _freshness_hz_lines(rows: list[Sample]) -> list[str]:
    """Render WebSocket update rates derived from mean inter-arrival time."""
    streams = [
        s for s in rows
        if s.name == "inter-arrival" and not s.error and s.timings_ns
    ]
    if not streams:
        return []
    label_w = max(max(len(s.category) for s in streams), len("stream"))
    header = f"{'stream':<{label_w}}  {'updates/sec':>12}  {'mean gap':>10}"
    out = [
        "",
        "Price freshness — sustained WS update rate (= 1 / mean inter-arrival):",
        header,
        "-" * len(header),
    ]
    for s in streams:
        mean_us = s.percentiles_us()["mean"]
        hz = 1_000_000.0 / mean_us if mean_us and mean_us > 0 else float("inf")
        out.append(
            f"{s.category:<{label_w}}  {hz:>9.1f} Hz  {_fmt_us(mean_us):>10}"
        )
    return out


def format_report(samples: Iterable[Sample]) -> str:
    """Render the benchmark table."""
    rows = list(samples)
    if not rows:
        return "(no samples)\n"
    name_w = max(len(f"{s.category} :: {s.name}") for s in rows)
    name_w = max(name_w, len("function"))
    header = (
        f"{'function':<{name_w}}  "
        f"{'iters':>5}  "
        f"{'mean':>9}  "
        f"{'p50':>9}  "
        f"{'p95':>9}  "
        f"{'p99':>9}  "
        f"{'max':>9}  "
        f"notes"
    )
    lines: list[str] = ["", header, "-" * len(header)]
    last_cat = ""
    for s in rows:
        if s.category != last_cat:
            if last_cat:
                lines.append("")  # blank line between categories
            last_cat = s.category
        label = f"{s.category} :: {s.name}"
        if s.error:
            lines.append(f"{label:<{name_w}}  ERROR: {s.error}")
            continue
        p = s.percentiles_us()
        lines.append(
            f"{label:<{name_w}}  "
            f"{s.n_iterations:>5d}  "
            f"{_fmt_us(p['mean']):>9}  "
            f"{_fmt_us(p['p50']):>9}  "
            f"{_fmt_us(p['p95']):>9}  "
            f"{_fmt_us(p['p99']):>9}  "
            f"{_fmt_us(p['max']):>9}  "
            f"{s.notes}"
        )
    lines.extend(_freshness_hz_lines(rows))
    lines.append("")
    lines.append(
        "Legend: timings are wall-clock; 1µs = 1000ns. "
        "p50/p95/p99 are sample percentiles. Price-freshness Hz = "
        "1 / mean inter-arrival (sustained WS update rate)."
    )
    return "\n".join(lines) + "\n"


def print_report(samples: Iterable[Sample]) -> None:
    sys.stdout.write(format_report(samples))


def write_text_report(
    samples: Iterable[Sample],
    path: Path,
    *,
    preamble: str = "",
) -> None:
    """Write the same human-readable table to *path*, with optional preamble."""
    body = format_report(samples)
    path.write_text(preamble + body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Latency benchmark for the realtime trading agent API.",
    )
    parser.add_argument(
        "--profile",
        choices=("local", "binance", "alpaca", "all"),
        default="local",
        help=(
            "local: in-memory + synthetic provider only (no network). "
            "binance: also probe live Binance REST. "
            "alpaca: also probe live Alpaca REST (needs API keys). "
            "all: local + binance + alpaca."
        ),
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help=(
            "Symbols to benchmark. Defaults: BTCUSDT for binance/local, "
            "SPY for alpaca. The first symbol is used for the per-symbol "
            "tests; multiple symbols increase the in-memory-suite cost "
            "linearly (only the first is benchmarked currently)."
        ),
    )
    parser.add_argument(
        "--iterations", type=int, default=2000,
        help="Iterations per microbenchmark (default 2000).",
    )
    parser.add_argument(
        "--warmup", type=int, default=50,
        help="Warmup iterations before timing starts (default 50).",
    )
    parser.add_argument(
        "--live-iterations", type=int, default=20,
        help=(
            "Iterations for network probes (binance/alpaca). REST calls "
            "are slow + rate-limited; keep this small. Default 20."
        ),
    )
    parser.add_argument(
        "--live-warmup", type=int, default=2,
        help="Warmup iterations for network probes (default 2).",
    )
    parser.add_argument(
        "--ws-seconds", type=float, default=15.0,
        help=(
            "WS-probe window (seconds) for each live profile. The "
            "binance leg subscribes to @aggTrade + @bookTicker in "
            "parallel; the alpaca leg subscribes to trades + quotes "
            "in one connection. 0 disables WS probes."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help=(
            "Optional path to save the same human-readable table that is "
            "printed to the terminal (plus a small header). Use any "
            "extension you like — .txt / .log / .md all work."
        ),
    )
    parser.add_argument(
        "--no-paper-executor", action="store_true",
        help=(
            "alpaca profile only: skip the AlpacaPaperExecutor read-only "
            "REST probes (get_cash / get_positions / tick / etc.). By "
            "default these run because they are read-only."
        ),
    )
    parser.add_argument(
        "--include-paper-mutations", action="store_true",
        help=(
            "alpaca profile only: ALSO submit + cancel real paper orders "
            "to measure the mutating AlpacaPaperExecutor REST surface. "
            "Orders are far-from-market limits that should not fill, and "
            "each submit is paired with a cancel, but they DO flash in "
            "your Alpaca paper dashboard. reset() is NOT probed (it would "
            "close all paper positions). Off by default."
        ),
    )
    args = parser.parse_args(argv)

    profile = args.profile
    if args.symbols:
        symbols = args.symbols
    elif profile == "alpaca":
        symbols = ["SPY"]
    else:
        symbols = ["BTCUSDT"]
    symbol = symbols[0]

    print(f"[bench] profile={profile} symbol={symbol} "
          f"iterations={args.iterations} warmup={args.warmup}")

    all_samples: list[Sample] = []

    print("[bench] running in-memory suite (no network) ...")
    all_samples.extend(
        _bench_in_memory_suite(symbol, args.iterations, args.warmup)
    )

    # Reuse Binance's offset for Alpaca, which has no public time endpoint.
    binance_offset_ms: float = 0.0

    if profile in ("binance", "all"):
        print(f"[bench] running Binance REST + WS probe "
              f"(REST iterations={args.live_iterations}, "
              f"WS window={args.ws_seconds:.0f}s) ...")
        try:
            all_samples.extend(
                _bench_binance(
                    symbol, args.live_iterations, args.live_warmup,
                    ws_seconds=args.ws_seconds,
                )
            )
            if args.ws_seconds > 0:
                try:
                    from openfinai_pipeline.realtime.data_providers.binance import (
                        BinanceProvider as _BP,
                    )
                    binance_offset_ms = _measure_binance_clock_offset_ms(_BP())
                except Exception:
                    binance_offset_ms = 0.0
        except Exception as exc:  # network errors should not abort the report
            all_samples.append(Sample(
                category="BinanceProvider",
                name="(setup failed)",
                n_iterations=0,
                error=f"{type(exc).__name__}: {exc}",
            ))

    if profile in ("alpaca", "all"):
        alpaca_symbol = "SPY" if symbol == "BTCUSDT" else symbol
        print(f"[bench] running Alpaca REST + WS probe "
              f"(REST iterations={args.live_iterations}, "
              f"WS window={args.ws_seconds:.0f}s, "
              f"symbol={alpaca_symbol}, "
              f"paper_executor={'on' if not args.no_paper_executor else 'off'}, "
              f"paper_mutations={'on' if args.include_paper_mutations else 'off'}) ...")
        if args.include_paper_mutations:
            print(
                "[bench] WARNING: --include-paper-mutations will submit + cancel "
                "real orders on your Alpaca paper account."
            )
        try:
            all_samples.extend(
                _bench_alpaca(
                    alpaca_symbol,
                    args.live_iterations,
                    args.live_warmup,
                    include_paper_executor=not args.no_paper_executor,
                    include_paper_mutations=args.include_paper_mutations,
                    ws_seconds=args.ws_seconds,
                    binance_offset_ms=binance_offset_ms,
                )
            )
        except Exception as exc:
            all_samples.append(Sample(
                category="AlpacaProvider",
                name="(setup failed)",
                n_iterations=0,
                error=f"{type(exc).__name__}: {exc}",
            ))

    print_report(all_samples)

    if args.out is not None:
        preamble = (
            f"# Realtime trading latency benchmark\n"
            f"# generated: {datetime.now(timezone.utc).isoformat()}\n"
            f"# profile:   {profile}\n"
            f"# symbol:    {symbol}\n"
            f"# iterations={args.iterations} warmup={args.warmup} "
            f"live_iterations={args.live_iterations} "
            f"live_warmup={args.live_warmup}\n"
            f"#\n"
            f"# AGENT-ACTION LATENCY: the per-action cost an agent pays is the\n"
            f"#   'task.step[real provider, internal_paper]' row — framework side:\n"
            f"#   obs build + executor tick/submit + PnL recompute. The price it\n"
            f"#   reads is served from the WS-fed buffer in ~µs (REST fires only\n"
            f"#   when buffer staleness exceeds the TTL). It excludes the agent's\n"
            f"#   own decision compute (its train.py step()), which is unbounded.\n"
            f"# NO ACTION-FREQUENCY GUARD: the in-house internal_paper executor does\n"
            f"#   not pace the gym loop — an agent may submit actions as fast as it\n"
            f"#   wants (e.g. many per second), even though new bars arrive only\n"
            f"#   ~once per minute. Price freshness is bounded by WS push (see the\n"
            f"#   'Price freshness' Hz block below), not by the step rate. The\n"
            f"#   alpaca_paper / live-REST paths instead hit broker rate limits +\n"
            f"#   RTH (reactive 429 retry), not framework-side pacing.\n"
        )
        write_text_report(all_samples, args.out, preamble=preamble)
        print(f"[bench] wrote {args.out}")

    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(main())
