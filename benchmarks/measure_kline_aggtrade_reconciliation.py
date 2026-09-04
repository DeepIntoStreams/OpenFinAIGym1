"""Reconcile locally aggregated Binance ``@aggTrade`` data with 1s klines.

The script compares OHLCV and taker-buy volume after a grace period, excluding
partial boundary bars. Kline trade count is reported but not compared because
an aggregate trade may represent multiple trades.

Run::

    conda activate openfingym
    python benchmarks/measure_kline_aggtrade_reconciliation.py \\
        --symbol BTCUSDT --ws-seconds 60
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
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
except Exception:
    pass


logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
logger = logging.getLogger("reconcile")
logger.setLevel(logging.INFO)


# WS runner


class Reconciler:
    """Run both WS subs in parallel, accumulate trades + closed-bars."""

    def __init__(self, symbol: str, duration_s: float) -> None:
        self.symbol = symbol
        self.duration_s = duration_s
        self._lock = threading.Lock()
        # bucket_open_ms -> list[(T_ms, price, qty, is_buyer_maker)]
        self._buckets: dict[int, list[tuple[int, float, float, bool]]] = (
            defaultdict(list)
        )
        # list of dicts: kline closed-bar payloads
        self._closed_bars: list[dict[str, Any]] = []
        # diagnostic counts
        self._aggtrade_count = 0
        self._kline_msg_count = 0
        # threads
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self._threads = [
            threading.Thread(
                target=self._run_stream, args=("aggTrade", self._on_aggtrade),
                daemon=True, name="r-aggT",
            ),
            threading.Thread(
                target=self._run_stream, args=("kline_1s", self._on_kline),
                daemon=True, name="r-kline",
            ),
        ]
        for t in self._threads:
            t.start()

    def join(self, timeout: float | None = None) -> None:
        for t in self._threads:
            t.join(timeout=timeout)

    def _run_stream(self, stream_suffix: str, on_msg) -> None:
        import websockets

        url = (
            f"wss://stream.binance.com:9443/ws/"
            f"{self.symbol.lower()}@{stream_suffix}"
        )
        deadline = time.monotonic() + self.duration_s

        async def _consume() -> None:
            async with websockets.connect(url, ping_interval=20) as ws:
                while time.monotonic() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as exc:
                        logger.warning(
                            "[%s] recv error: %s", stream_suffix, exc
                        )
                        return
                    try:
                        on_msg(json.loads(raw))
                    except Exception:
                        logger.debug(
                            "[%s] parse error", stream_suffix, exc_info=True
                        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_consume())
        finally:
            loop.close()

    def _on_aggtrade(self, msg: dict[str, Any]) -> None:
        T = int(msg["T"])
        bucket_open = (T // 1000) * 1000
        price = float(msg["p"])
        qty = float(msg["q"])
        is_maker_buy = bool(msg["m"])
        with self._lock:
            self._buckets[bucket_open].append((T, price, qty, is_maker_buy))
            self._aggtrade_count += 1

    def _on_kline(self, msg: dict[str, Any]) -> None:
        k = msg.get("k", {})
        self._kline_msg_count += 1
        if not k.get("x"):
            return  # in-progress update, ignore
        with self._lock:
            self._closed_bars.append(
                {
                    "t": int(k["t"]),
                    "T": int(k["T"]),
                    "o": float(k["o"]),
                    "h": float(k["h"]),
                    "l": float(k["l"]),
                    "c": float(k["c"]),
                    "v": float(k["v"]),
                    "q": float(k.get("q", "0")),
                    "V": float(k.get("V", "0")),
                    "Q": float(k.get("Q", "0")),
                    "n": int(k.get("n", 0)),
                }
            )

    # scoring

    def score(
        self, grace_s: float = 1.5, rel_tol: float = 1e-8
    ) -> dict[str, Any]:
        # Let late aggTrades for the most recent closed bars settle in.
        time.sleep(grace_s)
        with self._lock:
            bars = sorted(self._closed_bars, key=lambda b: b["t"])
            buckets = {
                k: sorted(v, key=lambda r: r[0])
                for k, v in self._buckets.items()
            }
            aggtrade_count = self._aggtrade_count
            kline_msg_count = self._kline_msg_count

        if len(bars) < 3:
            return {
                "scored_bars": 0,
                "aggtrade_count": aggtrade_count,
                "kline_msg_count": kline_msg_count,
                "note": "fewer than 3 closed kline bars; nothing to score",
            }

        # Skip first + last to avoid boundary partials.
        scoreable_bars = bars[1:-1]

        rows: list[dict[str, Any]] = []
        match_counts = {f: 0 for f in ("o", "h", "l", "c", "v", "q", "V", "Q")}
        divergences: list[dict[str, Any]] = []
        # Two distinct empty-bucket categories — kline reports closed bars
        # even for bar-windows with zero trades, so an empty bucket with
        # kline.n == 0 is *expected* (we don't synthesise empty bars).
        # An empty bucket with kline.n > 0 means we genuinely missed
        # aggTrades for that bar — a real correctness failure.
        empty_kline_n_zero = 0    # genuinely empty bars; not divergences
        empty_kline_n_positive = 0  # we missed aggTrades → real divergence

        for bar in scoreable_bars:
            bucket = buckets.get(bar["t"], [])
            if not bucket:
                if bar["n"] == 0:
                    empty_kline_n_zero += 1
                    status = "empty_no_trades_in_bar"
                else:
                    empty_kline_n_positive += 1
                    status = "missing_aggtrades"
                rows.append(
                    {
                        "bar_t": bar["t"],
                        "status": status,
                        "kline_n": bar["n"],
                    }
                )
                continue
            local_o = bucket[0][1]
            local_c = bucket[-1][1]
            local_h = max(t[1] for t in bucket)
            local_l = min(t[1] for t in bucket)
            local_v = sum(t[2] for t in bucket)
            local_q = sum(t[1] * t[2] for t in bucket)
            local_V = sum(t[2] for t in bucket if not t[3])
            local_Q = sum(t[1] * t[2] for t in bucket if not t[3])
            local_n_agg = len(bucket)

            def _eq(a: float, b: float) -> bool:
                if a == b:
                    return True
                denom = max(abs(a), abs(b))
                if denom == 0.0:
                    return True
                return abs(a - b) / denom < rel_tol

            checks = {
                "o": _eq(bar["o"], local_o),
                "h": _eq(bar["h"], local_h),
                "l": _eq(bar["l"], local_l),
                "c": _eq(bar["c"], local_c),
                "v": _eq(bar["v"], local_v),
                "q": _eq(bar["q"], local_q),
                "V": _eq(bar["V"], local_V),
                "Q": _eq(bar["Q"], local_Q),
            }
            for f, ok in checks.items():
                if ok:
                    match_counts[f] += 1

            row = {
                "bar_t": bar["t"],
                "status": (
                    "match" if all(checks.values()) else "divergent"
                ),
                "checks": checks,
                "kline": {
                    f: bar[f] for f in ("o", "h", "l", "c", "v", "q", "V", "Q")
                },
                "local": {
                    "o": local_o, "h": local_h, "l": local_l, "c": local_c,
                    "v": local_v, "q": local_q, "V": local_V, "Q": local_Q,
                },
                "kline_n": bar["n"],
                "local_aggtrade_count": local_n_agg,
            }
            rows.append(row)
            if not all(checks.values()):
                divergences.append(row)

        # Denominator for match rate is "bars where trades happened" —
        # the genuinely-empty bars (kline.n == 0) are excluded because
        # they're trivially-equal-by-vacuous-truth and skew the rate down.
        n_scored = len(scoreable_bars)
        n_with_trades = n_scored - empty_kline_n_zero
        return {
            "scored_bars": n_scored,
            "scored_bars_with_trades": n_with_trades,
            "aggtrade_count": aggtrade_count,
            "kline_msg_count": kline_msg_count,
            "empty_kline_n_zero": empty_kline_n_zero,
            "empty_kline_n_positive": empty_kline_n_positive,
            "match_rate_per_field_of_traded": {
                f: (match_counts[f] / n_with_trades if n_with_trades else 0.0)
                for f in match_counts
            },
            "divergence_count": len(divergences),
            "divergences": divergences,
            "rows": rows,
        }


# main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--ws-seconds", type=float, default=60.0)
    p.add_argument("--grace-seconds", type=float, default=1.5)
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

    logger.info(
        "starting reconciliation: symbol=%s window=%.0fs grace=%.1fs",
        args.symbol, args.ws_seconds, args.grace_seconds,
    )
    r = Reconciler(args.symbol, args.ws_seconds)
    r.start()
    r.join(timeout=args.ws_seconds + 5.0)
    result = r.score(grace_s=args.grace_seconds)

    out_root = Path(args.output_dir) / (
        "reconcile_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 78)
    print(
        f"aggTrade ↔ kline_1s reconciliation  symbol={args.symbol}  "
        f"window={args.ws_seconds:.0f}s"
    )
    print(f"output: {out_root}")
    print("=" * 78)
    print(
        f"\naggTrade messages received: {result['aggtrade_count']}"
    )
    print(f"kline messages received (any kind): {result['kline_msg_count']}")
    print(f"bars scored (excl. boundary): {result['scored_bars']}")
    if result.get("note"):
        print(f"NOTE: {result['note']}")
        return 0
    print(
        f"  of which: bars with kline.n>0 (actually-traded): "
        f"{result['scored_bars_with_trades']}"
    )
    print(
        f"            bars with kline.n==0 (genuinely empty, excluded): "
        f"{result['empty_kline_n_zero']}"
    )
    if result["empty_kline_n_positive"]:
        print(
            f"  *** MISSING-AGGTRADE BARS (kline.n>0 but no local data): "
            f"{result['empty_kline_n_positive']} ***"
        )

    denom = result["scored_bars_with_trades"]
    print(
        f"\nper-field match rate (over {denom} actually-traded bars):"
    )
    for f, rate in result["match_rate_per_field_of_traded"].items():
        print(f"  {f}: {rate * 100:.2f}% ({int(rate * denom)}/{denom})")
    print(f"\ntotal divergent bars (any field): {result['divergence_count']}")
    if result["divergence_count"] > 0:
        print("first 3 divergent bars:")
        for d in result["divergences"][:3]:
            print(json.dumps(d, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
