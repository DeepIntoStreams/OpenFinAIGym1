"""Compare Binance, Finnhub, and Alpaca WebSocket stream latency.

The benchmark records message cadence, aligned one-way latency where an event
timestamp exists, and Binance aggregate-trade ID gaps. Provider clock drift can
affect cross-provider one-way comparisons. Credentials are loaded from ``.env``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Load repository credentials when available.
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
except Exception:
    pass

from openfinai_pipeline.realtime.data_providers.binance import BinanceProvider  # noqa: E402

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
logger = logging.getLogger("prov_streams")
logger.setLevel(logging.INFO)


def measure_binance_offset(samples: int = 5) -> float:
    """Return ``binance_server_epoch_ms - local_epoch_ms`` via Cristian's."""
    provider = BinanceProvider()
    offsets_ms: list[float] = []
    for _ in range(samples):
        t0 = time.time() * 1000.0
        resp = provider._get("/api/v3/time")  # type: ignore[attr-defined]
        t1 = time.time() * 1000.0
        server_ms = float(resp["serverTime"])
        offsets_ms.append(server_ms - (t0 + t1) / 2.0)
        time.sleep(0.1)
    return statistics.median(offsets_ms)


class StreamRecorder:
    """Record a provider-specific WebSocket stream for ``duration_s``."""

    def __init__(
        self,
        name: str,
        duration_s: float,
        clock_offset_ms: float,
        connect_factory: Callable[[asyncio.AbstractEventLoop], Any],
        on_message: Callable[[dict[str, Any]], list[dict[str, Any]]],
        maxlen: int = 200_000,
    ) -> None:
        self.name = name
        self.duration_s = duration_s
        self.clock_offset_ms = clock_offset_ms
        self.rows: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.first_msg_event = threading.Event()
        self.connect_factory = connect_factory
        self.on_message = on_message
        self.error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"rec-{name}"
        )

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _run(self) -> None:
        deadline = time.monotonic() + self.duration_s
        last_recv_aligned_ms: float | None = None
        msg_idx = 0

        async def _consume() -> None:
            nonlocal last_recv_aligned_ms, msg_idx
            try:
                ws = await self.connect_factory(asyncio.get_event_loop())
            except Exception as exc:
                self.error = exc
                logger.warning("[%s] connect failed: %s", self.name, exc)
                return
            try:
                while time.monotonic() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as exc:
                        self.error = exc
                        logger.warning("[%s] recv error: %s", self.name, exc)
                        return
                    t_recv_local_ms = time.time() * 1000.0
                    t_recv_aligned_ms = t_recv_local_ms + self.clock_offset_ms
                    try:
                        rows_from_msg = self.on_message(json.loads(raw))
                    except Exception as exc:
                        logger.debug(
                            "[%s] parse error: %s — raw=%r", self.name, exc, raw[:200]
                        )
                        continue
                    for partial in rows_from_msg:
                        row = {
                            "stream": self.name,
                            "i": msg_idx,
                            "recv_local_epoch_ms": t_recv_local_ms,
                            "recv_aligned_epoch_ms": t_recv_aligned_ms,
                            **partial,
                        }
                        if "event_E_ms" in row and row["event_E_ms"] is not None:
                            row["one_way_ms"] = t_recv_aligned_ms - float(
                                row["event_E_ms"]
                            )
                        if last_recv_aligned_ms is not None:
                            row["inter_arrival_ms"] = (
                                t_recv_aligned_ms - last_recv_aligned_ms
                            )
                        last_recv_aligned_ms = t_recv_aligned_ms
                        self.rows.append(row)
                        msg_idx += 1
                        if not self.first_msg_event.is_set():
                            self.first_msg_event.set()
            finally:
                try:
                    await ws.close()
                except Exception:
                    pass

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_consume())
        finally:
            loop.close()


def _binance_factory(stream: str, symbol: str):
    async def _connect(_loop: asyncio.AbstractEventLoop):
        import websockets

        url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@{stream}"
        return await websockets.connect(url, ping_interval=20)

    return _connect


def _binance_aggtrade_parser(msg: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "event_E_ms": msg.get("E"),
            "trade_T_ms": msg.get("T"),
            "agg_id": msg.get("a"),
            "first_trade_id": msg.get("f"),
            "last_trade_id": msg.get("l"),
            "price": float(msg.get("p", "nan")),
            "qty": float(msg.get("q", "nan")),
        }
    ]


def _binance_kline_parser(msg: dict[str, Any]) -> list[dict[str, Any]]:
    k = msg.get("k", {})
    return [
        {
            "event_E_ms": msg.get("E"),
            "bar_open_ms": k.get("t"),
            "bar_close_ms": k.get("T"),
            "is_closed": bool(k.get("x", False)),
            "price": float(k.get("c", "nan")),
        }
    ]


def _finnhub_factory(symbols: list[str], api_key: str):
    async def _connect(_loop: asyncio.AbstractEventLoop):
        import websockets

        url = f"wss://ws.finnhub.io?token={api_key}"
        ws = await websockets.connect(url, ping_interval=20)
        for s in symbols:
            await ws.send(json.dumps({"type": "subscribe", "symbol": s}))
        return ws

    return _connect


def _finnhub_parser(msg: dict[str, Any]) -> list[dict[str, Any]]:
    if msg.get("type") != "trade":
        return []
    rows = []
    for item in msg.get("data") or []:
        rows.append(
            {
                "event_E_ms": item.get("t"),
                "symbol": item.get("s"),
                "price": float(item.get("p", "nan")),
                "qty": float(item.get("v", "nan")) if item.get("v") is not None else None,
            }
        )
    return rows


def _alpaca_factory(symbols: list[str], api_key: str, api_secret: str):
    async def _connect(_loop: asyncio.AbstractEventLoop):
        import websockets

        ws = await websockets.connect(
            "wss://stream.data.alpaca.markets/v2/iex", ping_interval=20
        )
        await ws.send(
            json.dumps({"action": "auth", "key": api_key, "secret": api_secret})
        )
        await ws.recv()
        await ws.send(json.dumps({"action": "subscribe", "trades": symbols}))
        await ws.recv()
        return ws

    return _connect


def _alpaca_parser(msg: Any) -> list[dict[str, Any]]:
    if not isinstance(msg, list):
        msg = [msg]
    rows = []
    for item in msg:
        if not isinstance(item, dict) or item.get("T") != "t":
            continue
        ts_str = item.get("t")
        ts_ms = None
        if isinstance(ts_str, str):
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                ts_ms = ts.timestamp() * 1000.0
            except Exception:
                ts_ms = None
        rows.append(
            {
                "event_E_ms": ts_ms,
                "symbol": item.get("S"),
                "price": float(item.get("p", "nan")),
                "qty": float(item.get("s", "nan")) if item.get("s") is not None else None,
            }
        )
    return rows


def _dist(values: list[float]) -> dict[str, float] | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "p50": statistics.median(s),
        "p95": s[max(0, int(0.95 * (len(s) - 1)))],
        "p99": s[max(0, int(0.99 * (len(s) - 1)))],
        "min": s[0],
        "max": s[-1],
    }


def _aggtrade_gaps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [r.get("agg_id") for r in rows if r.get("agg_id") is not None]
    if len(ids) < 2:
        return {"sequence_observed": False}
    spans = ids[-1] - ids[0] + 1
    received = len(ids)
    missing = spans - received
    gap_sizes: list[int] = []
    for prev, curr in zip(ids, ids[1:]):
        diff = curr - prev
        if diff > 1:
            gap_sizes.append(diff - 1)
    return {
        "sequence_observed": True,
        "first_agg_id": ids[0],
        "last_agg_id": ids[-1],
        "agg_id_span": spans,
        "messages_received": received,
        "missing_agg_ids": missing,
        "missing_pct": (missing / spans * 100.0) if spans else 0.0,
        "gap_count": len(gap_sizes),
        "gap_size": _dist([float(g) for g in gap_sizes]) if gap_sizes else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crypto-symbol", default="BTCUSDT")
    p.add_argument("--equity-symbol", default="AAPL")
    p.add_argument(
        "--finnhub-symbols",
        default="BINANCE:BTCUSDT,AAPL",
        help=(
            "comma-separated Finnhub symbols. Including BINANCE:BTCUSDT "
            "lets us compare Finnhub's relay latency against the direct "
            "Binance @aggTrade stream apples-to-apples."
        ),
    )
    p.add_argument("--ws-seconds", type=float, default=60.0)
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
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="stream names to skip (binance_aggtrade, binance_kline_1s, "
        "finnhub_trade, alpaca_trade)",
    )
    args = p.parse_args()

    logger.info("syncing local clock with Binance /api/v3/time …")
    offset_ms = measure_binance_offset()
    logger.info("local clock offset (binance - local): %+.1f ms", offset_ms)

    finnhub_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    alpaca_key = os.environ.get("ALPACA_API_KEY", "").strip()
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    # Strip inline comments (.env preserves trailing "# foo")
    for var in (finnhub_key, alpaca_key, alpaca_secret):
        pass
    finnhub_key = finnhub_key.split("#", 1)[0].strip()
    alpaca_key = alpaca_key.split("#", 1)[0].strip()
    alpaca_secret = alpaca_secret.split("#", 1)[0].strip()

    recorders: list[StreamRecorder] = []
    if "binance_aggtrade" not in args.skip:
        recorders.append(
            StreamRecorder(
                "binance_aggtrade",
                args.ws_seconds,
                offset_ms,
                _binance_factory("aggTrade", args.crypto_symbol),
                _binance_aggtrade_parser,
            )
        )
    if "binance_kline_1s" not in args.skip:
        recorders.append(
            StreamRecorder(
                "binance_kline_1s",
                args.ws_seconds,
                offset_ms,
                _binance_factory("kline_1s", args.crypto_symbol),
                _binance_kline_parser,
            )
        )
    if "finnhub_trade" not in args.skip and finnhub_key:
        finnhub_syms = [
            s.strip() for s in args.finnhub_symbols.split(",") if s.strip()
        ]
        recorders.append(
            StreamRecorder(
                "finnhub_trade",
                args.ws_seconds,
                offset_ms,
                _finnhub_factory(finnhub_syms, finnhub_key),
                _finnhub_parser,
            )
        )
    elif "finnhub_trade" not in args.skip:
        logger.warning("FINNHUB_API_KEY missing; skipping finnhub_trade")
    if "alpaca_trade" not in args.skip and alpaca_key and alpaca_secret:
        recorders.append(
            StreamRecorder(
                "alpaca_trade",
                args.ws_seconds,
                offset_ms,
                _alpaca_factory([args.equity_symbol], alpaca_key, alpaca_secret),
                _alpaca_parser,
            )
        )
    elif "alpaca_trade" not in args.skip:
        logger.warning(
            "ALPACA_API_KEY/ALPACA_SECRET_KEY missing; skipping alpaca_trade"
        )

    logger.info(
        "running %d recorders for %.1fs in parallel …",
        len(recorders),
        args.ws_seconds,
    )
    for r in recorders:
        r.start()
    for r in recorders:
        r.join(timeout=args.ws_seconds + 10.0)

    out_root = Path(args.output_dir) / (
        "provider_streams_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    for r in recorders:
        rows = list(r.rows)
        (out_root / f"{r.name}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        s: dict[str, Any] = {
            "count": len(rows),
            "msgs_per_s": (len(rows) / args.ws_seconds) if rows else 0.0,
            "one_way_ms": _dist(
                [row["one_way_ms"] for row in rows if "one_way_ms" in row]
            ),
            "inter_arrival_ms": _dist(
                [row["inter_arrival_ms"] for row in rows if "inter_arrival_ms" in row]
            ),
            "error": repr(r.error) if r.error else None,
        }
        if r.name == "binance_aggtrade":
            s["gaps"] = _aggtrade_gaps(rows)
        if r.name in ("finnhub_trade", "alpaca_trade"):
            per_sym: dict[str, dict[str, Any]] = {}
            for row in rows:
                sym = row.get("symbol") or "?"
                bucket = per_sym.setdefault(
                    sym, {"count": 0, "ow": [], "ia": []}
                )
                bucket["count"] += 1
                if "one_way_ms" in row:
                    bucket["ow"].append(row["one_way_ms"])
                if "inter_arrival_ms" in row:
                    bucket["ia"].append(row["inter_arrival_ms"])
            s["per_symbol"] = {
                sym: {
                    "count": v["count"],
                    "one_way_ms": _dist(v["ow"]),
                    "inter_arrival_ms": _dist(v["ia"]),
                }
                for sym, v in per_sym.items()
            }
        summaries[r.name] = s

    summary = {
        "config": vars(args),
        "clock_offset_ms_binance": offset_ms,
        "summary_per_stream": summaries,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 78)
    print(
        f"Provider-stream comparison  "
        f"crypto={args.crypto_symbol}  equity={args.equity_symbol}  "
        f"window={args.ws_seconds:.0f}s"
    )
    print(f"binance clock offset (server - local): {offset_ms:+.1f} ms")
    print(f"output: {out_root}")
    print("=" * 78)

    print(
        f"\n{'stream':<22}{'count':>7}{'msg/s':>9}"
        f"{'1-way p50':>12}{'1-way p95':>12}{'IA p50':>10}{'IA p95':>10}  (ms)"
    )
    for name, s in summaries.items():
        ow = s.get("one_way_ms") or {}
        ia = s.get("inter_arrival_ms") or {}
        print(
            f"{name:<22}{s['count']:>7}{s['msgs_per_s']:>9.2f}"
            f"{ow.get('p50', float('nan')):>12.1f}"
            f"{ow.get('p95', float('nan')):>12.1f}"
            f"{ia.get('p50', float('nan')):>10.1f}"
            f"{ia.get('p95', float('nan')):>10.1f}"
        )

    for prov in ("finnhub_trade", "alpaca_trade"):
        if prov in summaries and summaries[prov].get("per_symbol"):
            print(f"\n{prov} per-symbol:")
            for sym, ps in summaries[prov]["per_symbol"].items():
                ow = ps["one_way_ms"] or {}
                ia = ps["inter_arrival_ms"] or {}
                print(
                    f"  {sym:<22}  count={ps['count']:>5}  "
                    f"1-way p50={ow.get('p50', float('nan')):>7.1f}ms  "
                    f"IA p50={ia.get('p50', float('nan')):>7.1f}ms"
                )

    if "binance_aggtrade" in summaries:
        g = summaries["binance_aggtrade"].get("gaps", {})
        print("\nBinance @aggTrade aggregate-trade-ID continuity:")
        if not g.get("sequence_observed"):
            print("  insufficient messages to assess (need >=2)")
        else:
            print(
                f"  first id={g['first_agg_id']}  last id={g['last_agg_id']}  "
                f"span={g['agg_id_span']:,}  received={g['messages_received']:,}"
            )
            print(
                f"  missing_agg_ids={g['missing_agg_ids']:,}  "
                f"({g['missing_pct']:.3f}% of span)"
            )
            print(f"  gap_count={g['gap_count']}")
            if g.get("gap_size"):
                gs = g["gap_size"]
                print(
                    f"  gap_size (ids):  p50={gs['p50']:.0f}  p95={gs['p95']:.0f}  "
                    f"max={gs['max']:.0f}"
                )

    errors = {n: s["error"] for n, s in summaries.items() if s.get("error")}
    if errors:
        print("\nERRORS encountered:")
        for n, e in errors.items():
            print(f"  {n}: {e}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
