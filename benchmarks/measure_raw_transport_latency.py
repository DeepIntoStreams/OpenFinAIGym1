"""Compare Binance REST round-trip and WebSocket one-way latency.

The WebSocket measurements cover kline, trade, and book-ticker streams. Local
receive times are aligned with Binance server time through ``/api/v3/time``.
Raw JSONL and a summary are written below the configured output directory.
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from openfinai_pipeline.realtime.data_providers.binance import BinanceProvider  # noqa: E402

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
logger = logging.getLogger("raw_latency")
logger.setLevel(logging.INFO)


def measure_clock_offset(provider: BinanceProvider, samples: int = 5) -> float:
    """Return the median server-minus-local clock offset."""
    offsets_ms: list[float] = []
    for _ in range(samples):
        t0 = time.time() * 1000.0
        resp = provider._get("/api/v3/time")  # type: ignore[attr-defined]
        t1 = time.time() * 1000.0
        server_ms = float(resp["serverTime"])
        local_mid = (t0 + t1) / 2.0
        offsets_ms.append(server_ms - local_mid)
        time.sleep(0.1)
    return statistics.median(offsets_ms)


def measure_rest_rtt(
    provider: BinanceProvider,
    symbol: str,
    n_calls: int,
    inter_call_sleep: float,
) -> list[dict[str, Any]]:
    """Record sequential ``get_current_price`` round-trip times."""
    rows: list[dict[str, Any]] = []
    for i in range(n_calls):
        t0 = time.monotonic()
        snap = provider.get_current_price(symbol)
        t1 = time.monotonic()
        rows.append(
            {
                "kind": "rest",
                "i": i,
                "rtt_ms": (t1 - t0) * 1000.0,
                "recv_epoch_ms": time.time() * 1000.0,
                "price": snap.price,
                "bar_open_ms": int(snap.timestamp.timestamp() * 1000.0),
            }
        )
        if inter_call_sleep > 0:
            time.sleep(inter_call_sleep)
    return rows


class StreamRecorder:
    """Subscribe to a single Binance WS stream and record per-message data."""

    def __init__(
        self,
        symbol: str,
        stream_suffix: str,
        duration_s: float,
        clock_offset_ms: float,
        maxlen: int = 50_000,
    ) -> None:
        self.symbol = symbol
        self.stream_suffix = stream_suffix
        self.duration_s = duration_s
        self.clock_offset_ms = clock_offset_ms
        self.rows: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"raw-{stream_suffix}"
        )

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    @property
    def done(self) -> threading.Event:
        return self._done

    def _run(self) -> None:
        import websockets

        url = (
            f"wss://stream.binance.com:9443/ws/"
            f"{self.symbol.lower()}@{self.stream_suffix}"
        )
        deadline = time.monotonic() + self.duration_s
        last_recv_aligned_ms: float | None = None
        msg_idx = 0

        async def _consume() -> None:
            nonlocal last_recv_aligned_ms, msg_idx
            async with websockets.connect(url, ping_interval=20) as ws:
                while time.monotonic() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as exc:
                        logger.warning(
                            "ws error (%s): %s", self.stream_suffix, exc
                        )
                        return
                    t_recv_local_ms = time.time() * 1000.0
                    t_recv_aligned_ms = t_recv_local_ms + self.clock_offset_ms
                    msg = json.loads(raw)

                    E = msg.get("E")
                    row: dict[str, Any] = {
                        "kind": self.stream_suffix,
                        "i": msg_idx,
                        "recv_local_epoch_ms": t_recv_local_ms,
                        "recv_aligned_epoch_ms": t_recv_aligned_ms,
                        "event_E_ms": E,
                    }
                    if E is not None:
                        row["one_way_ms"] = t_recv_aligned_ms - float(E)
                    if last_recv_aligned_ms is not None:
                        row["inter_arrival_ms"] = (
                            t_recv_aligned_ms - last_recv_aligned_ms
                        )
                    last_recv_aligned_ms = t_recv_aligned_ms

                    if self.stream_suffix.startswith("kline"):
                        k = msg.get("k", {})
                        row["price"] = float(k.get("c", "nan"))
                        row["bar_open_ms"] = int(k.get("t", 0))
                        row["bar_closed"] = bool(k.get("x", False))
                    elif self.stream_suffix == "trade":
                        row["price"] = float(msg.get("p", "nan"))
                        row["trade_T_ms"] = int(msg.get("T", 0))
                        if msg.get("T") is not None:
                            row["trade_to_recv_ms"] = (
                                t_recv_aligned_ms - float(msg["T"])
                            )
                    elif self.stream_suffix == "bookTicker":
                        row["best_bid"] = float(msg.get("b", "nan"))
                        row["best_ask"] = float(msg.get("a", "nan"))

                    self.rows.append(row)
                    msg_idx += 1

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_consume())
        finally:
            loop.close()
            self._done.set()


def _dist(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    s = sorted(values)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "median": statistics.median(s),
        "p95": s[max(0, int(0.95 * (len(s) - 1)))],
        "p99": s[max(0, int(0.99 * (len(s) - 1)))],
        "min": s[0],
        "max": s[-1],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--rest-calls", type=int, default=60)
    p.add_argument("--rest-sleep", type=float, default=0.2)
    p.add_argument("--ws-seconds", type=float, default=30.0)
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

    provider = BinanceProvider()

    logger.info("syncing clock with Binance /api/v3/time …")
    offset_ms = measure_clock_offset(provider)
    logger.info("clock offset (server - local): %+.1f ms", offset_ms)

    logger.info(
        "starting 3 WS recorders for %.1fs each in parallel …", args.ws_seconds
    )
    recorders = [
        StreamRecorder(args.symbol, "kline_1m", args.ws_seconds, offset_ms),
        StreamRecorder(args.symbol, "trade", args.ws_seconds, offset_ms),
        StreamRecorder(args.symbol, "bookTicker", args.ws_seconds, offset_ms),
    ]
    for r in recorders:
        r.start()
    # Let all WebSocket handshakes finish before the REST probe starts.
    time.sleep(1.0)

    logger.info(
        "starting %d REST calls (sleep %.2fs between) …",
        args.rest_calls,
        args.rest_sleep,
    )
    rest_rows = measure_rest_rtt(
        provider, args.symbol, args.rest_calls, args.rest_sleep
    )
    rest_duration = (
        rest_rows[-1]["recv_epoch_ms"] - rest_rows[0]["recv_epoch_ms"]
    ) / 1000.0
    logger.info(
        "REST done in %.1fs; waiting for WS recorders to finish …",
        rest_duration,
    )

    for r in recorders:
        r.join(timeout=args.ws_seconds + 5.0)

    out_root = Path(args.output_dir) / (
        "raw_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "rest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rest_rows) + "\n", encoding="utf-8"
    )
    for r in recorders:
        rows = list(r.rows)
        (out_root / f"ws_{r.stream_suffix}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    rest_rtts = [r["rtt_ms"] for r in rest_rows]
    ws_summaries: dict[str, dict[str, Any]] = {}
    for r in recorders:
        rows = list(r.rows)
        ws_summaries[r.stream_suffix] = {
            "count": len(rows),
            "duration_s": args.ws_seconds,
            "msgs_per_s": (len(rows) / args.ws_seconds) if rows else 0.0,
            "one_way_ms": _dist(
                [row["one_way_ms"] for row in rows if "one_way_ms" in row]
            ),
            "inter_arrival_ms": _dist(
                [row["inter_arrival_ms"] for row in rows if "inter_arrival_ms" in row]
            ),
        }

    summary = {
        "config": {
            "symbol": args.symbol,
            "rest_calls": args.rest_calls,
            "rest_sleep": args.rest_sleep,
            "ws_seconds": args.ws_seconds,
        },
        "clock_offset_ms": offset_ms,
        "rest_rtt_ms": _dist(rest_rtts),
        "ws": ws_summaries,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 76)
    print(f"Raw transport latency — symbol={args.symbol}")
    print(f"clock_offset (server - local): {offset_ms:+.1f} ms")
    print(f"output dir: {out_root}")
    print("=" * 76)

    rt = summary["rest_rtt_ms"]
    print(f"\nREST round-trip latency (GET /api/v3/klines, in-process REST)")
    print(f"  n={rt['n']}")
    print(
        f"  mean={rt['mean']:.1f}  median={rt['median']:.1f}  "
        f"p95={rt['p95']:.1f}  p99={rt['p99']:.1f}  "
        f"min={rt['min']:.1f}  max={rt['max']:.1f}  (ms)"
    )

    print(f"\nWS one-way latency  (server-side event time E → local arrival)")
    print(f"{'stream':<14}{'count':>7}{'msg/s':>9}"
          f"{'mean':>9}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}  (ms)")
    for suffix, s in ws_summaries.items():
        ow = s["one_way_ms"]
        if not ow:
            continue
        print(
            f"{suffix:<14}{s['count']:>7}{s['msgs_per_s']:>9.2f}"
            f"{ow['mean']:>9.1f}{ow['median']:>8.1f}"
            f"{ow['p95']:>8.1f}{ow['p99']:>8.1f}{ow['max']:>8.1f}"
        )

    print(f"\nWS inter-arrival cadence  (push frequency, smaller = faster updates)")
    print(f"{'stream':<14}{'mean':>9}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}  (ms)")
    for suffix, s in ws_summaries.items():
        ia = s["inter_arrival_ms"]
        if not ia:
            continue
        print(
            f"{suffix:<14}{ia['mean']:>9.1f}{ia['median']:>8.1f}"
            f"{ia['p95']:>8.1f}{ia['p99']:>8.1f}{ia['max']:>8.1f}"
        )

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
