"""Compare agent-observed Binance prices from REST and WebSocket paths.

An independent trade stream supplies the reference for price lag, price gap,
and order-book staleness metrics. Raw rows and a summary are written to the
realtime diagnostics output directory.

Run::

    conda activate openfingym
    python benchmarks/measure_realtime_latency.py --symbol BTCUSDT \\
        --steps 30 --inter-step 1.5

"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make src/ + repo root importable. The framework's contracts module
# does `from data.knowledge_base.rewards import ...`, where ``data``
# lives at the repo root, not under ``src/``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from openfinai_pipeline.realtime.data_providers.binance import BinanceProvider  # noqa: E402
from openfinai_pipeline.realtime.tasks.realtime_trading_task import (  # noqa: E402
    RealtimeTradingTask,
)
from openfinai_pipeline.settings import TradingConfig  # noqa: E402


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("latency_bench")
logger.setLevel(logging.INFO)


# Non-invasive WS toggle


class RestOnlyBinanceProvider(BinanceProvider):
    """BinanceProvider with WS disabled. Pure REST hot path."""

    def supports_websocket(self) -> bool:  # noqa: PLR6301
        return False


# Reference fetcher — independent ground truth


class ReferenceFetcher:
    """Independent WS subscriptions for trades + depth top-of-book.

    Two parallel streams so the framework's own WS connections don't
    compete:
        - ``<symbol>@trade``     — every executed trade (tick-level)
        - ``<symbol>@bookTicker`` — every best bid/ask change

    Trades feed ``_trades`` as ``(t_recv_mono, t_recv_epoch, price)``.
    bookTicker feeds ``_books`` as
    ``(t_recv_mono, t_recv_epoch, best_bid, best_ask)``.
    """

    def __init__(self, symbol: str, maxlen: int = 200_000) -> None:
        self.symbol = symbol
        self._trades: deque[tuple[float, float, float]] = deque(maxlen=maxlen)
        self._books: deque[tuple[float, float, float, float]] = deque(maxlen=maxlen)
        self._trades_lock = threading.Lock()
        self._books_lock = threading.Lock()
        self._stop = threading.Event()
        self._first_trade = threading.Event()
        self._first_book = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self._threads = [
            threading.Thread(
                target=self._run_stream,
                args=("trade", self._on_trade),
                daemon=True,
                name=f"refws-trade-{self.symbol}",
            ),
            threading.Thread(
                target=self._run_stream,
                args=("bookTicker", self._on_book),
                daemon=True,
                name=f"refws-book-{self.symbol}",
            ),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()

    def wait_warm(self, timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout
        if not self._first_trade.wait(max(0.0, deadline - time.monotonic())):
            return False
        if not self._first_book.wait(max(0.0, deadline - time.monotonic())):
            return False
        return True

    def _run_stream(self, stream_suffix: str, on_msg) -> None:
        import websockets

        url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@{stream_suffix}"

        async def _consume() -> None:
            async with websockets.connect(url, ping_interval=20) as ws:
                while not self._stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as exc:
                        logger.warning("reference ws error (%s): %s", stream_suffix, exc)
                        return
                    on_msg(json.loads(raw))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_consume())
        except Exception:
            logger.warning("reference fetcher %s stopped", stream_suffix, exc_info=True)
        finally:
            loop.close()

    def _on_trade(self, msg: dict[str, Any]) -> None:
        price_raw = msg.get("p")
        if price_raw is None:
            return
        t_mono = time.monotonic()
        t_epoch = time.time()
        with self._trades_lock:
            self._trades.append((t_mono, t_epoch, float(price_raw)))
        if not self._first_trade.is_set():
            self._first_trade.set()

    def _on_book(self, msg: dict[str, Any]) -> None:
        bid_raw = msg.get("b")
        ask_raw = msg.get("a")
        if bid_raw is None or ask_raw is None:
            return
        t_mono = time.monotonic()
        t_epoch = time.time()
        with self._books_lock:
            self._books.append((t_mono, t_epoch, float(bid_raw), float(ask_raw)))
        if not self._first_book.is_set():
            self._first_book.set()

    # scoring helpers

    def latest_price_at(self, t_mono: float) -> float | None:
        with self._trades_lock:
            events = list(self._trades)
        for i in range(len(events) - 1, -1, -1):
            if events[i][0] <= t_mono:
                return events[i][2]
        return None

    def stale_lag(self, obs_price: float, t_mono: float) -> float | None:
        """Time since the world moved away from ``obs_price``, in seconds.

        Walks reference trades backward from ``t_mono``: find the
        latest tick with ``price == obs_price``. Everything between
        that tick and ``t_mono`` is at some other price (since by
        definition the latest match is the last). Lag = ``t_mono -
        time of first tick AFTER the latest match``. If no match
        found in window, returns ``None``. If the latest tick at
        ``t_mono`` is already a match, returns ``0`` (no staleness).
        """
        with self._trades_lock:
            events = list(self._trades)
        idx_latest_match = None
        for i in range(len(events) - 1, -1, -1):
            if events[i][0] > t_mono:
                continue
            if events[i][2] == obs_price:
                idx_latest_match = i
                break
        if idx_latest_match is None:
            return None
        # First tick strictly after the latest match, that's still <= t_mono.
        for j in range(idx_latest_match + 1, len(events)):
            if events[j][0] > t_mono:
                break
            if events[j][2] != obs_price:
                return t_mono - events[j][0]
        return 0.0  # latest known reference tick is still the obs price

    def latest_book_at(self, t_mono: float) -> tuple[float, float, float] | None:
        """Most-recent (bid, ask, wall_clock_mono) at or before ``t_mono``."""
        with self._books_lock:
            events = list(self._books)
        for i in range(len(events) - 1, -1, -1):
            if events[i][0] <= t_mono:
                return events[i][2], events[i][3], events[i][0]
        return None

    def book_match_age(
        self, obs_bid: float | None, obs_ask: float | None, t_mono: float
    ) -> float | None:
        """Seconds since reference last had (bid, ask) == (obs_bid, obs_ask).

        Matches on both sides exactly. Returns ``None`` if either side
        is missing or no historical match exists. ``0`` if the most
        recent reference book matches obs (i.e., obs is fresh).
        """
        if obs_bid is None or obs_ask is None:
            return None
        with self._books_lock:
            events = list(self._books)
        for i in range(len(events) - 1, -1, -1):
            if events[i][0] > t_mono:
                continue
            if events[i][2] == obs_bid and events[i][3] == obs_ask:
                # Walk forward: if the very next book differs, the world
                # has moved away from this state since events[i].
                return t_mono - events[i][0]
        return None

    def counts(self) -> tuple[int, int]:
        with self._trades_lock:
            t = len(self._trades)
        with self._books_lock:
            b = len(self._books)
        return t, b


# Benchmark driver


def _run_mode(
    *,
    mode_name: str,
    symbol: str,
    n_steps: int,
    warmup_steps: int,
    inter_step_sleep: float,
    reference: ReferenceFetcher,
    use_ws: bool,
) -> list[dict[str, Any]]:
    """Drive RealtimeTradingTask for ``n_steps`` and record per-step rows."""
    provider_cls = BinanceProvider if use_ws else RestOnlyBinanceProvider
    provider = provider_cls()
    trading_config = TradingConfig(execution_mode="internal_paper")
    task = RealtimeTradingTask(
        provider=provider,
        symbols=[symbol],
        trading_config=trading_config,
        context_resolutions=[{"interval": "1m", "bars": 30}],
        max_steps=n_steps + warmup_steps + 5,
        initial_capital=100_000.0,
    )
    logger.info(
        "mode=%s loading task (REST backfill + WS=%s)…",
        mode_name,
        "on" if use_ws else "off",
    )
    task.reset()

    rows: list[dict[str, Any]] = []
    hold_action = {"action": "hold", "symbol": symbol, "quantity": 0.0}

    total_steps = warmup_steps + n_steps
    for step_idx in range(total_steps):
        t_started = time.monotonic()
        obs, reward, done, info = task.step(hold_action)
        t_returned = time.monotonic()

        sym_obs = (obs or {}).get("symbols", {}).get(symbol, {})
        obs_price = sym_obs.get("price")
        obs_bar_ts = sym_obs.get("timestamp")
        order_book = sym_obs.get("order_book")
        obs_bid = getattr(order_book, "best_bid", None) if order_book else None
        obs_ask = getattr(order_book, "best_ask", None) if order_book else None

        ref_latest = reference.latest_price_at(t_returned)
        if obs_price is not None and ref_latest is not None and ref_latest != 0:
            price_gap_bps = (obs_price / ref_latest - 1.0) * 1.0e4
            in_sync = obs_price == ref_latest
        else:
            price_gap_bps = None
            in_sync = None
        stale_lag_s = (
            reference.stale_lag(obs_price, t_returned)
            if obs_price is not None
            else None
        )
        # If in sync, stale_lag should be 0; only report >0 cases as
        # meaningful "staleness" lag, but persist the raw value either way.

        latest_book = reference.latest_book_at(t_returned)
        ob_match_age_s = reference.book_match_age(obs_bid, obs_ask, t_returned)
        if latest_book and obs_bid is not None and obs_ask is not None:
            ref_bid, ref_ask, _ = latest_book
            ob_in_sync = (ref_bid == obs_bid) and (ref_ask == obs_ask)
        else:
            ob_in_sync = None

        rows.append(
            {
                "mode": mode_name,
                "step": step_idx,
                "is_warmup": step_idx < warmup_steps,
                "t_started_mono": t_started,
                "t_returned_mono": t_returned,
                "step_duration_s": t_returned - t_started,
                "obs_price": obs_price,
                "obs_bar_ts": obs_bar_ts,
                "ref_latest_price": ref_latest,
                "price_gap_bps": price_gap_bps,
                "in_sync": in_sync,
                "stale_lag_s": stale_lag_s,
                "order_book_present": order_book is not None,
                "obs_bid": obs_bid,
                "obs_ask": obs_ask,
                "ob_in_sync": ob_in_sync,
                "ob_match_age_s": ob_match_age_s,
                "reward": float(reward),
                "done": bool(done),
            }
        )

        if done:
            logger.info("mode=%s episode ended early at step=%d", mode_name, step_idx)
            break
        if inter_step_sleep > 0:
            time.sleep(inter_step_sleep)

    task.close()
    return rows


def _summarize(rows: list[dict[str, Any]], mode_name: str) -> dict[str, Any]:
    scored = [r for r in rows if not r["is_warmup"]]
    n = len(scored)
    if n == 0:
        return {"mode": mode_name, "n": 0}

    def _stats(values: list[float], *, ms: bool = False) -> dict[str, float]:
        if not values:
            return {}
        s = sorted(values)
        scale = 1000.0 if ms else 1.0
        return {
            "mean": statistics.fmean(s) * scale,
            "median": statistics.median(s) * scale,
            "p95": s[max(0, int(0.95 * (len(s) - 1)))] * scale,
            "p99": s[max(0, int(0.99 * (len(s) - 1)))] * scale,
            "max": s[-1] * scale,
            "min": s[0] * scale,
        }

    stale_lags = [
        r["stale_lag_s"]
        for r in scored
        if r["stale_lag_s"] is not None and r["stale_lag_s"] > 0
    ]
    gap = [abs(r["price_gap_bps"]) for r in scored if r["price_gap_bps"] is not None]
    dur = [r["step_duration_s"] for r in scored]
    ob_ages = [
        r["ob_match_age_s"]
        for r in scored
        if r["ob_match_age_s"] is not None
    ]
    ob_pct = sum(1 for r in scored if r["order_book_present"]) / n * 100.0
    sync_n = sum(1 for r in scored if r["in_sync"] is True)
    ob_sync_n = sum(1 for r in scored if r["ob_in_sync"] is True)
    ob_total = sum(1 for r in scored if r["ob_in_sync"] is not None)

    return {
        "mode": mode_name,
        "n": n,
        "in_sync_pct": sync_n / n * 100.0,
        "stale_lag_ms_when_stale": _stats(stale_lags, ms=True),
        "abs_price_gap_bps": _stats(gap),
        "step_duration_ms": _stats(dur, ms=True),
        "order_book_present_pct": ob_pct,
        "ob_in_sync_pct": (ob_sync_n / ob_total * 100.0) if ob_total else None,
        "ob_match_age_ms": _stats(ob_ages, ms=True),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--steps", type=int, default=30, help="scored steps per mode")
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="leading steps excluded from stats",
    )
    p.add_argument(
        "--inter-step",
        type=float,
        default=1.5,
        help="sleep seconds between consecutive step() calls",
    )
    p.add_argument(
        "--output-dir",
        default=str(
            _REPO_ROOT
            / "data"
            / "run_output"
            / "realtime_diagnostics"
            / "latency_benchmark"
        ),
    )
    args = p.parse_args()

    out_root = Path(args.output_dir) / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info("starting reference fetcher for %s…", args.symbol)
    ref = ReferenceFetcher(args.symbol)
    ref.start()
    if not ref.wait_warm(timeout=15.0):
        logger.error("reference fetcher never warmed up — aborting")
        return 2
    logger.info("reference fetcher live (trades + bookTicker warm)")

    summaries: list[dict[str, Any]] = []
    for mode_name, use_ws in (
        ("rest_only", False),
        ("rest_ws", True),
    ):
        rows = _run_mode(
            mode_name=mode_name,
            symbol=args.symbol,
            n_steps=args.steps,
            warmup_steps=args.warmup_steps,
            inter_step_sleep=args.inter_step,
            reference=ref,
            use_ws=use_ws,
        )
        (out_root / f"{mode_name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )
        summary = _summarize(rows, mode_name)
        summaries.append(summary)

    ref.stop()
    trade_n, book_n = ref.counts()
    (out_root / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "symbol": args.symbol,
                    "steps": args.steps,
                    "warmup_steps": args.warmup_steps,
                    "inter_step_sleep": args.inter_step,
                },
                "reference_trades_total": trade_n,
                "reference_books_total": book_n,
                "modes": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(f"Latency benchmark — symbol={args.symbol} steps={args.steps}/mode")
    print(f"reference trades={trade_n}  bookTicker events={book_n}")
    print(f"output dir: {out_root}")
    print("=" * 70)

    def _fmt(s: dict[str, float] | dict[str, Any], key: str = "median") -> str:
        return f"{s[key]:.1f}" if s else "—"

    for s in summaries:
        print(f"\n[{s['mode']}]  n={s['n']}")
        if s["n"] == 0:
            continue
        gap = s["abs_price_gap_bps"]
        dur = s["step_duration_ms"]
        ob_age = s["ob_match_age_ms"]
        stale = s["stale_lag_ms_when_stale"]
        print(f"  PRICE")
        print(f"    in_sync at step end: {s['in_sync_pct']:.1f}%")
        if stale:
            print(
                f"    stale_lag when stale (ms): "
                f"mean={stale['mean']:.1f}  median={stale['median']:.1f}  "
                f"p95={stale['p95']:.1f}  max={stale['max']:.1f}"
            )
        else:
            print("    stale_lag: no stale observations in window")
        if gap:
            print(
                f"    |price_gap| (bps): "
                f"mean={gap['mean']:.3f}  p95={gap['p95']:.3f}  "
                f"max={gap['max']:.3f}"
            )
        print(f"  ORDER BOOK")
        print(f"    present: {s['order_book_present_pct']:.1f}%")
        if s["ob_in_sync_pct"] is not None:
            print(f"    in_sync at step end: {s['ob_in_sync_pct']:.1f}%")
        if ob_age:
            print(
                f"    ob_match_age (ms): "
                f"mean={ob_age['mean']:.1f}  median={ob_age['median']:.1f}  "
                f"p95={ob_age['p95']:.1f}  max={ob_age['max']:.1f}"
            )
        print(f"  STEP")
        print(
            f"    duration (ms): "
            f"mean={dur['mean']:.1f}  median={dur['median']:.1f}  "
            f"p95={dur['p95']:.1f}  max={dur['max']:.1f}"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
