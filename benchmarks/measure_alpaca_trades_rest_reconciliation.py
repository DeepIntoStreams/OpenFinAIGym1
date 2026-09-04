"""Compare locally aggregated Alpaca WebSocket trades with REST minute bars.

Trades are bucketed by server timestamp and compared on OHLCV after dropping
partial boundary minutes. Divergences include condition-code diagnostics.
Requires ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY``.

Run::

    conda activate openfingym
    python benchmarks/measure_alpaca_trades_rest_reconciliation.py \\
        --symbols SPY,AAPL,QQQ,IWM --ws-seconds 3600

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
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
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

from openfinai_pipeline.realtime.data_providers.alpaca import AlpacaProvider  # noqa: E402

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
logger = logging.getLogger("alpaca_reconcile")
logger.setLevel(logging.INFO)


# WS recorder


class _TradeRecorder:
    """Stream Alpaca WS ``trades`` for ``symbols`` into per-minute buckets.

    ``minute_open_ms -> dict[symbol -> list[trade-tuple]]`` where each
    trade-tuple is ``(t_ms, price, size, conditions_tuple)``.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: list[str],
        duration_s: float,
    ) -> None:
        self.api_key = api_key.split("#", 1)[0].strip()
        self.api_secret = api_secret.split("#", 1)[0].strip()
        self.symbols = symbols
        self.duration_s = duration_s
        self._lock = threading.Lock()
        # minute_open_ms -> {symbol: [trades...]}
        self._buckets: dict[int, dict[str, list[tuple[int, float, float, tuple[str, ...]]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self._trade_count = 0
        self._conditions: Counter[str] = Counter()
        self._first_trade = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="alp-rec")

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def wait_first_trade(self, timeout: float = 30.0) -> bool:
        return self._first_trade.wait(timeout)

    @property
    def error(self) -> Exception | None:
        return self._error

    @property
    def trade_count(self) -> int:
        return self._trade_count

    @property
    def conditions(self) -> Counter[str]:
        with self._lock:
            return Counter(self._conditions)

    def snapshot(self) -> dict[int, dict[str, list[tuple[int, float, float, tuple[str, ...]]]]]:
        with self._lock:
            return {
                m: {s: list(trades) for s, trades in by_sym.items()}
                for m, by_sym in self._buckets.items()
            }

    def _on_msg(self, msg: dict[str, Any]) -> None:
        if msg.get("T") != "t":
            return  # auth/sub responses, error frames, non-trade messages
        symbol = msg.get("S")
        ts_str = msg.get("t")
        if symbol is None or ts_str is None:
            return
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            return
        ts_ms = int(ts.timestamp() * 1000)
        bucket_open = (ts_ms // 60_000) * 60_000
        price = float(msg["p"])
        size = float(msg["s"])
        conditions = tuple(msg.get("c") or [])
        with self._lock:
            self._buckets[bucket_open][symbol].append((ts_ms, price, size, conditions))
            self._trade_count += 1
            for c in conditions:
                self._conditions[c] += 1
        if not self._first_trade.is_set():
            self._first_trade.set()

    def _run(self) -> None:
        import websockets

        url = "wss://stream.data.alpaca.markets/v2/iex"

        async def _consume() -> None:
            async with websockets.connect(url, ping_interval=20) as ws:
                # Auth
                await ws.send(
                    json.dumps(
                        {"action": "auth", "key": self.api_key, "secret": self.api_secret}
                    )
                )
                # Two response frames typically: connect ack + auth ack.
                # Drain until we see auth status.
                for _ in range(4):
                    raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    payload = json.loads(raw)
                    if isinstance(payload, list):
                        for item in payload:
                            t = item.get("T")
                            if t == "error":
                                raise RuntimeError(
                                    f"Alpaca WS error: {item}"
                                )
                            if t == "success" and item.get("msg") == "authenticated":
                                break
                        else:
                            continue
                        break
                # Subscribe
                await ws.send(
                    json.dumps({"action": "subscribe", "trades": self.symbols})
                )
                # Drain subscribe ack
                await asyncio.wait_for(ws.recv(), timeout=10.0)

                deadline = time.monotonic() + self.duration_s
                while time.monotonic() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as exc:
                        self._error = exc
                        logger.warning("ws recv error: %s", exc)
                        return
                    payload = json.loads(raw)
                    if isinstance(payload, list):
                        for item in payload:
                            self._on_msg(item)
                    elif isinstance(payload, dict):
                        self._on_msg(payload)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_consume())
        except Exception as exc:
            self._error = exc
            logger.warning("alpaca recorder stopped: %s", exc, exc_info=True)
        finally:
            loop.close()


# Scoring


# Alpaca REST bars exclude ``I``-condition trades from OHLC but retain their
# volume. Keep the set narrow; reconciliation divergences surface new cases.
_NON_LAST_SALE_CODES: frozenset[str] = frozenset({"I"})


def _is_last_sale_eligible(conditions: tuple[str, ...]) -> bool:
    """A trade is last-sale-eligible iff none of its condition codes
    are in the non-last-sale set."""
    return not any(c in _NON_LAST_SALE_CODES for c in conditions)


def _local_ohlcv(
    trades: list[tuple[int, float, float, tuple[str, ...]]],
    *,
    filter_last_sale: bool = False,
) -> dict[str, Any]:
    """Aggregate a per-bucket trade list into OHLCV + condition histogram.

    When ``filter_last_sale`` is True, OHLC are computed only from
    last-sale-eligible trades while volume still sums *all* trades —
    matching the standard NASDAQ-UTP / Alpaca bar-aggregation rule.
    """
    trades_sorted = sorted(trades, key=lambda t: t[0])
    v = sum(t[2] for t in trades_sorted)
    if filter_last_sale:
        ohlc_trades = [t for t in trades_sorted if _is_last_sale_eligible(t[3])]
    else:
        ohlc_trades = trades_sorted
    cond_counter: Counter[str] = Counter()
    for t in trades_sorted:
        for cd in t[3]:
            cond_counter[cd] += 1
    if not ohlc_trades:
        # Every trade in the bucket was non-last-sale → REST would
        # carry forward the previous bar's close. We can't compute it
        # in isolation; surface NaN so the caller knows.
        return {
            "o": float("nan"), "h": float("nan"),
            "l": float("nan"), "c": float("nan"),
            "v": v,
            "trade_count": len(trades_sorted),
            "ohlc_trade_count": 0,
            "conditions": dict(cond_counter),
        }
    o = ohlc_trades[0][1]
    c = ohlc_trades[-1][1]
    h = max(t[1] for t in ohlc_trades)
    low = min(t[1] for t in ohlc_trades)
    return {
        "o": o, "h": h, "l": low, "c": c, "v": v,
        "trade_count": len(trades_sorted),
        "ohlc_trade_count": len(ohlc_trades),
        "conditions": dict(cond_counter),
    }


def _eq(a: float, b: float, rel_tol: float = 1e-6) -> bool:
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    if denom == 0.0:
        return True
    return abs(a - b) / denom < rel_tol


# Main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--symbols",
        default="SPY",
        help="comma-separated symbols. Subscribed in a single WS "
             "connection; scored independently and aggregated.",
    )
    p.add_argument(
        "--symbol",
        default=None,
        help="(legacy) single-symbol alias for --symbols",
    )
    p.add_argument("--ws-seconds", type=float, default=360.0,
                   help="recording window in seconds")
    p.add_argument("--rest-grace-seconds", type=float, default=5.0,
                   help="sleep after stopping WS before fetching REST, so "
                        "the just-closed bar is finalised server-side")
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

    api_key = os.environ.get("ALPACA_API_KEY", "").split("#", 1)[0].strip()
    api_secret = os.environ.get("ALPACA_SECRET_KEY", "").split("#", 1)[0].strip()
    if not api_key or not api_secret:
        logger.error(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY missing in env / .env"
        )
        return 2

    # Resolve symbols: --symbol legacy single beats --symbols default if set.
    if args.symbol:
        symbols = [args.symbol.strip()]
    else:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("no symbols specified")
        return 6

    logger.info(
        "starting Alpaca WS trades recorder: symbols=%s window=%.0fs",
        symbols, args.ws_seconds,
    )
    recorder = _TradeRecorder(api_key, api_secret, symbols, args.ws_seconds)
    recorder.start()
    warm = recorder.wait_first_trade(timeout=30.0)
    if not warm:
        logger.warning(
            "no trades arrived in 30 s of warmup — Alpaca IEX may be quiet "
            "for these symbols right now (US market 14:30-21:00 UTC); "
            "continuing anyway"
        )
    recorder.join(timeout=args.ws_seconds + 10.0)
    if recorder.error is not None:
        logger.error("recorder ended with error: %s", recorder.error)
        return 3
    if recorder.trade_count == 0:
        logger.error("no trades captured — cannot reconcile")
        return 4

    logger.info(
        "captured %d trades across %d symbols; sleeping %.1fs grace "
        "then fetching REST bars",
        recorder.trade_count, len(symbols), args.rest_grace_seconds,
    )
    time.sleep(args.rest_grace_seconds)

    buckets = recorder.snapshot()
    if len(buckets) < 3:
        logger.error(
            "captured trades span only %d minute-buckets — need >=3 "
            "for boundary-safe scoring", len(buckets),
        )
        return 5

    sorted_buckets = sorted(buckets.keys())
    scoreable = sorted_buckets[1:-1]   # boundary-trim
    start_ts = datetime.fromtimestamp(scoreable[0] / 1000, tz=timezone.utc)
    end_ts = datetime.fromtimestamp(
        (scoreable[-1] + 60_000) / 1000, tz=timezone.utc
    )

    provider = AlpacaProvider()

    # Per-symbol scoring
    per_symbol_summary: dict[str, dict[str, Any]] = {}
    # Aggregate counters across all symbols
    agg_match_raw = {f: 0 for f in ("o", "h", "l", "c", "v")}
    agg_match_filtered = {f: 0 for f in ("o", "h", "l", "c", "v")}
    agg_n_with_trades = 0
    agg_divergent_count = 0

    for sym in symbols:
        try:
            rest_bars = provider.get_bars(sym, "1m", start_ts, end_ts)
        except Exception as exc:
            logger.warning(
                "REST get_bars failed for %s: %s — skipping symbol",
                sym, exc,
            )
            per_symbol_summary[sym] = {"error": repr(exc)}
            continue
        rest_map = {
            int(b.timestamp.timestamp() * 1000): b for b in rest_bars
        }

        rows: list[dict[str, Any]] = []
        match_counts_raw = {f: 0 for f in ("o", "h", "l", "c", "v")}
        match_counts_filtered = {f: 0 for f in ("o", "h", "l", "c", "v")}
        n_with_local_trades = 0
        divergent: list[dict[str, Any]] = []

        for bucket_open in scoreable:
            by_sym = buckets.get(bucket_open, {})
            trades = by_sym.get(sym, [])
            rest_bar = rest_map.get(bucket_open)
            row: dict[str, Any] = {"bucket_open_ms": bucket_open}
            if not trades and rest_bar is None:
                row["status"] = "empty_both"
                rows.append(row)
                continue
            if not trades and rest_bar is not None:
                row["status"] = "missing_local_trades"
                row["rest"] = {
                    f: getattr(rest_bar, f)
                    for f in ("open", "high", "low", "close", "volume")
                }
                rows.append(row)
                continue
            if trades and rest_bar is None:
                row["status"] = "missing_rest_bar"
                row["local_raw"] = _local_ohlcv(trades, filter_last_sale=False)
                row["local_filtered"] = _local_ohlcv(trades, filter_last_sale=True)
                rows.append(row)
                continue
            n_with_local_trades += 1
            local_raw = _local_ohlcv(trades, filter_last_sale=False)
            local_filtered = _local_ohlcv(trades, filter_last_sale=True)
            rest_ohlcv = {
                "o": float(rest_bar.open or rest_bar.price),
                "h": float(rest_bar.high or rest_bar.price),
                "l": float(rest_bar.low or rest_bar.price),
                "c": float(rest_bar.close or rest_bar.price),
                "v": float(rest_bar.volume or 0.0),
            }
            checks_raw = {
                f: _eq(local_raw[f], rest_ohlcv[f])
                for f in match_counts_raw
            }
            checks_filtered = {
                f: _eq(local_filtered[f], rest_ohlcv[f])
                for f in match_counts_filtered
            }
            for f, ok in checks_raw.items():
                if ok:
                    match_counts_raw[f] += 1
            for f, ok in checks_filtered.items():
                if ok:
                    match_counts_filtered[f] += 1
            row["status"] = (
                "match" if all(checks_filtered.values()) else "divergent"
            )
            row["checks_raw"] = checks_raw
            row["checks_filtered"] = checks_filtered
            row["local_raw"] = local_raw
            row["local_filtered"] = local_filtered
            row["rest"] = rest_ohlcv
            if not all(checks_filtered.values()):
                # Diagnostic dump: for each divergent bar, attach the
                # full per-trade list so we can inspect which condition
                # codes accompanied the trade that landed on the
                # mismatched OHLC value. Cheap because divergences are
                # rare; cuts the need for a separate re-run to debug.
                row["divergent_trades"] = [
                    {"t_ms": t[0], "price": t[1], "size": t[2],
                     "conditions": list(t[3])}
                    for t in trades
                ]
                divergent.append(row)
            rows.append(row)

        per_symbol_summary[sym] = {
            "scored_with_local_trades": n_with_local_trades,
            "match_rate_per_field_raw": {
                f: (match_counts_raw[f] / n_with_local_trades) if n_with_local_trades else None
                for f in match_counts_raw
            },
            "match_rate_per_field_filtered": {
                f: (match_counts_filtered[f] / n_with_local_trades) if n_with_local_trades else None
                for f in match_counts_filtered
            },
            "divergent_count_after_filter": len(divergent),
            "rows": rows,
            "divergent_rows": divergent,
        }
        for f in agg_match_raw:
            agg_match_raw[f] += match_counts_raw[f]
        for f in agg_match_filtered:
            agg_match_filtered[f] += match_counts_filtered[f]
        agg_n_with_trades += n_with_local_trades
        agg_divergent_count += len(divergent)

    summary = {
        "config": vars(args),
        "symbols": symbols,
        "trade_count": recorder.trade_count,
        "captured_buckets": len(sorted_buckets),
        "scored_buckets": len(scoreable),
        "agg_scored_with_local_trades": agg_n_with_trades,
        "agg_match_rate_per_field_raw": {
            f: (agg_match_raw[f] / agg_n_with_trades) if agg_n_with_trades else None
            for f in agg_match_raw
        },
        "agg_match_rate_per_field_filtered": {
            f: (agg_match_filtered[f] / agg_n_with_trades) if agg_n_with_trades else None
            for f in agg_match_filtered
        },
        "agg_divergent_count_after_filter": agg_divergent_count,
        "wire_condition_histogram": dict(recorder.conditions.most_common()),
        "non_last_sale_codes_filtered": sorted(_NON_LAST_SALE_CODES),
        "per_symbol": per_symbol_summary,
    }

    out_root = Path(args.output_dir) / (
        "alpaca_reconcile_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "result.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 78)
    print(
        f"Alpaca WS trades ↔ REST bars reconciliation  "
        f"symbols={','.join(symbols)}  window={args.ws_seconds:.0f}s"
    )
    print(f"output: {out_root}")
    print("=" * 78)
    print(f"\ntotal trades captured: {recorder.trade_count}")
    print(f"buckets captured:      {summary['captured_buckets']}")
    print(f"bars scored / symbol (excl. boundary): {summary['scored_buckets']}")
    print(f"aggregate bars with local trades:      {agg_n_with_trades}")
    print(f"wire condition histogram (top 12): "
          f"{recorder.conditions.most_common(12)}")

    print("\n──── per-symbol match rate (FILTERED vs REST) ──────────────────")
    print(f"{'symbol':<8}{'scored':>8}{'o':>9}{'h':>9}{'l':>9}{'c':>9}"
          f"{'v':>9}{'div':>6}")
    for sym in symbols:
        ps = per_symbol_summary.get(sym, {})
        if "error" in ps:
            print(f"{sym:<8}  ERROR: {ps['error']}")
            continue
        n = ps["scored_with_local_trades"]
        rates = ps["match_rate_per_field_filtered"]
        if n == 0:
            print(f"{sym:<8}{n:>8}   (no scoreable bars)")
            continue
        row = [
            f"{rates[f]*100:>8.2f}%" if rates[f] is not None else "      -"
            for f in ("o", "h", "l", "c", "v")
        ]
        print(f"{sym:<8}{n:>8}{row[0]:>9}{row[1]:>9}{row[2]:>9}{row[3]:>9}"
              f"{row[4]:>9}{ps['divergent_count_after_filter']:>6}")

    if agg_n_with_trades == 0:
        print("\nno scoreable bars across any symbol; cannot compute "
              "aggregate match rate.")
        return 0
    print("\n──── aggregate match rate (all symbols, all bars) ──────────────")
    print("local UNFILTERED vs REST:")
    for f, rate in summary["agg_match_rate_per_field_raw"].items():
        if rate is None:
            continue
        print(f"  {f}: {rate*100:6.2f}%  "
              f"({int(rate * agg_n_with_trades)}/{agg_n_with_trades})")
    print("local FILTERED (last-sale-eligibility) vs REST:")
    for f, rate in summary["agg_match_rate_per_field_filtered"].items():
        if rate is None:
            continue
        print(f"  {f}: {rate*100:6.2f}%  "
              f"({int(rate * agg_n_with_trades)}/{agg_n_with_trades})")
    print(f"\naggregate divergent bars after filtering: "
          f"{agg_divergent_count}")

    if agg_divergent_count:
        print("\nfirst 3 divergent rows across symbols (truncated dump):")
        all_div = []
        for sym in symbols:
            ps = per_symbol_summary.get(sym, {})
            for r in ps.get("divergent_rows", []):
                all_div.append((sym, r))
        for sym, r in all_div[:3]:
            keep = {
                k: v for k, v in r.items()
                if k not in ("local_raw", "local_filtered")
            }
            print(f"[{sym}]")
            print(json.dumps(keep, indent=2, default=str))
            for label in ("local_raw", "local_filtered"):
                if label in r:
                    lc = dict(r[label])
                    lc.pop("conditions", None)
                    print(f"  {label} (conditions hidden): {lc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
