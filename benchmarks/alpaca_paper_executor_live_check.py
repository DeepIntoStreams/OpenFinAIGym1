"""Exercise :class:`AlpacaPaperExecutor` against Alpaca's paper API.

WARNING: this script submits and cancels real paper-account orders, then calls
``reset()`` to flatten the account. It compares broker REST state with the
executor's local mirror. Market-fill checks require an open market; use
``--skip-fills`` to omit them. Alpaca credentials are loaded from ``.env``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Load credentials before constructing provider clients.
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
except Exception:  # pragma: no cover - .env is optional if env is preset
    pass

from openfinai_pipeline.realtime.data_providers.alpaca import (  # noqa: E402
    AlpacaProvider,
)
from openfinai_pipeline.realtime.execution.alpaca_paper import (  # noqa: E402
    AlpacaPaperExecutor,
)
from openfinai_pipeline.realtime.execution.base import (  # noqa: E402
    ActionVerb,
    OrderIntent,
    OrderStatus,
    OrderType,
    TimeInForce,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("alpaca_live_test")
logger.setLevel(logging.INFO)

_PAPER_BASE = "https://paper-api.alpaca.markets"


def _hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


_LATENCIES: dict[str, list[float]] = {}


def _record_latency(label: str, seconds: float) -> None:
    _LATENCIES.setdefault(label, []).append(seconds * 1000.0)


def _print_latency_summary() -> None:
    """Render the write-path timings gathered across the order steps."""
    if not _LATENCIES:
        return
    _hr("LATENCY (write-path, wall-clock)")
    print(f"  {'operation':<34} {'n':>3} {'mean':>9} {'min':>9} {'max':>9}")
    for label, ms in _LATENCIES.items():
        vals = sorted(ms)
        mean = sum(vals) / len(vals)
        print(
            f"  {label:<34} {len(vals):>3} {mean:>7.1f}ms "
            f"{vals[0]:>7.1f}ms {vals[-1]:>7.1f}ms"
        )
    print(
        "\n  submit[*] RTT is the blocking POST an async-submit refactor would "
        "remove from the\n  gym-loop hot path. 'WS provisional->filled gap' is "
        "the additional latency until the\n  local model is broker-accurate "
        "(NOT removed by async submit)."
    )


class _AlpacaProbe:
    """Read broker state independently of the executor's client."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._s = requests.Session()
        self._s.headers.update(
            {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
        )

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        r = self._s.get(
            f"{_PAPER_BASE}/v2/orders/{order_id}", timeout=15
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def get_positions(self) -> list[dict[str, Any]]:
        r = self._s.get(f"{_PAPER_BASE}/v2/positions", timeout=15)
        r.raise_for_status()
        out = r.json()
        return out if isinstance(out, list) else []

    def get_open_orders(self) -> list[dict[str, Any]]:
        r = self._s.get(
            f"{_PAPER_BASE}/v2/orders",
            params={"status": "open", "limit": 500},
            timeout=15,
        )
        r.raise_for_status()
        out = r.json()
        return out if isinstance(out, list) else []


def _poll_order_until(
    probe: _AlpacaProbe,
    order_id: str,
    target_statuses: Iterable[str],
    timeout: float,
) -> dict[str, Any] | None:
    """Poll Alpaca REST until the order reaches one of ``target_statuses``."""
    targets = set(target_statuses)
    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = probe.get_order(order_id)
        if last is not None and last.get("status") in targets:
            return last
        time.sleep(0.3)
    return last


def _wait_log_status(
    ex: AlpacaPaperExecutor,
    order_id: str,
    target_statuses: Iterable[str],
    timeout: float,
):
    """Poll the executor's reconciled trade log for a target status."""
    targets = set(target_statuses)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for rep in ex.get_trade_log():
            if rep.order_id == order_id and rep.status in targets:
                return rep
        time.sleep(0.2)
    return None


def _wait_log_status_timed(
    ex: AlpacaPaperExecutor,
    order_id: str,
    target_statuses: Iterable[str],
    timeout: float,
):
    """Tight-poll and return ``(report, elapsed_seconds)``."""
    targets = set(target_statuses)
    t0 = time.perf_counter()
    deadline = t0 + timeout
    while time.perf_counter() < deadline:
        for rep in ex.get_trade_log():
            if rep.order_id == order_id and rep.status in targets:
                return rep, time.perf_counter() - t0
        time.sleep(0.001)
    return None, None


def _reference_price(provider: AlpacaProvider, symbol: str) -> float:
    """Return the latest price, falling back to the NBBO midpoint."""
    try:
        snap = provider.get_current_price(symbol)
        if snap.price and snap.price > 0:
            return float(snap.price)
    except Exception as exc:
        logger.info("get_current_price failed (%s); trying NBBO mid", exc)
    ob = provider.get_order_book(symbol)
    mid = (ob.best_bid + ob.best_ask) / 2.0
    if mid <= 0:
        raise RuntimeError(f"could not determine a reference price for {symbol}")
    return float(mid)


def _check_reads(ex: AlpacaPaperExecutor) -> None:
    _hr("STEP 2 — portfolio reads (local model, seeded at construction)")
    cash = ex.get_cash()
    positions = ex.get_positions()
    pending = ex.get_pending_orders()
    print(f"  cash            = {cash}")
    print(f"  positions       = {positions}")
    print(f"  pending_orders  = {pending}")
    if cash is None:
        _fail("get_cash() returned None — account read did not seed cash")
    else:
        _ok(f"cash read back: {cash}")
    if positions == {}:
        _ok("positions empty (clean start)")
    else:
        _warn(f"positions not empty at start: {positions}")


def _check_market_order(
    ex: AlpacaPaperExecutor,
    probe: _AlpacaProbe,
    *,
    side: str,
    symbol: str,
    qty: float,
    price: float,
    fill_timeout: float,
) -> None:
    label = side.upper()
    _hr(f"market {label} {qty} {symbol} (ref price {price:.4f})")
    intent = OrderIntent(
        action=side,
        symbol=symbol,
        quantity=qty,
        order_type=OrderType.MARKET,
        tif=TimeInForce.IOC,
    )
    t0 = time.perf_counter()
    result = ex.submit(intent, market_price=price, step=1)
    _record_latency(f"submit[market {side}] RTT", time.perf_counter() - t0)
    print(f"  submit() -> kind={result.kind!r}")
    if result.kind == "rejected":
        _fail(f"market {label} rejected: {result.rejection.reason}")
        return
    if result.fill is None:
        _fail(f"market {label} returned kind={result.kind} with no fill")
        return
    order_id = result.fill.order_id
    print(
        f"  provisional booking: status={result.fill.status} "
        f"px={result.fill.executed_price} order_id={order_id}"
    )
    if result.fill.status == OrderStatus.PROVISIONAL:
        _ok("submit() returned synchronously with a PROVISIONAL booking")

    # Poll locally before the slower broker request to isolate WS latency.
    rep, gap = _wait_log_status_timed(
        ex, order_id, {OrderStatus.FILLED}, timeout=fill_timeout
    )
    if rep is not None and gap is not None:
        _record_latency("WS provisional->filled gap", gap)
        _ok(
            f"local model upgraded PROVISIONAL -> FILLED via WS in "
            f"{gap * 1000.0:.0f}ms (executed_price={rep.executed_price})"
        )
    else:
        _warn(
            "local model still PROVISIONAL (WS fill push not observed — "
            "expected if the fill beat the WS subscription; broker state "
            "below is authoritative)"
        )

    broker = _poll_order_until(
        probe, order_id, {"filled"}, timeout=fill_timeout
    )
    if broker is not None and broker.get("status") == "filled":
        _ok(
            f"ALPACA confirms FILLED: filled_qty={broker.get('filled_qty')} "
            f"filled_avg_price={broker.get('filled_avg_price')}"
        )
    else:
        status = broker.get("status") if broker else "<not found>"
        _warn(
            f"Alpaca did not report 'filled' within {fill_timeout:.0f}s "
            f"(status={status}); market may be illiquid/closed"
        )
    print(f"  positions after {label}: {ex.get_positions()}  cash={ex.get_cash()}")


def _check_limit_and_cancel(
    ex: AlpacaPaperExecutor,
    probe: _AlpacaProbe,
    *,
    symbol: str,
    qty: float,
    price: float,
    offset_pct: float,
) -> None:
    limit_price = round(price * (1.0 - offset_pct), 2)
    _hr(
        f"limit BUY {qty} {symbol} @ {limit_price} "
        f"({offset_pct*100:.0f}% below {price:.4f}; should rest, not fill)"
    )
    intent = OrderIntent(
        action=ActionVerb.BUY,
        symbol=symbol,
        quantity=qty,
        order_type=OrderType.LIMIT,
        limit_price=limit_price,
        tif=TimeInForce.GTC,
    )
    t0 = time.perf_counter()
    result = ex.submit(intent, market_price=price, step=2)
    _record_latency("submit[limit] RTT", time.perf_counter() - t0)
    print(f"  submit() -> kind={result.kind!r}")
    if result.kind != "accepted" or result.accepted is None:
        if result.kind == "rejected":
            _fail(f"limit order rejected: {result.rejection.reason}")
        else:
            _fail(f"limit order unexpected kind={result.kind}")
        return
    order_id = result.accepted.order_id
    _ok(f"limit order accepted (order_id={order_id})")

    pend_ids = [p.order_id for p in ex.get_pending_orders()]
    if order_id in pend_ids:
        _ok("present in executor.get_pending_orders()")
    else:
        _warn(f"not in local pending queue: {pend_ids}")

    broker = _poll_order_until(
        probe,
        order_id,
        {"new", "accepted", "held", "pending_new", "partially_filled"},
        timeout=10.0,
    )
    if broker is not None:
        _ok(
            f"ALPACA confirms resting order: status={broker.get('status')} "
            f"type={broker.get('type')} limit_price={broker.get('limit_price')}"
        )
    else:
        _warn("Alpaca did not return the limit order on REST lookup")

    print("  cancelling ...")
    t0 = time.perf_counter()
    cancel = ex.cancel(order_id)
    _record_latency("cancel RTT", time.perf_counter() - t0)
    print(f"  cancel() -> kind={cancel.kind!r}")
    if cancel.kind == "cancelled":
        _ok("executor.cancel() returned cancelled")
    else:
        _fail(f"cancel rejected: {cancel.rejection.reason if cancel.rejection else '?'}")

    broker = _poll_order_until(
        probe, order_id, {"canceled", "cancelled"}, timeout=10.0
    )
    if broker is not None and str(broker.get("status")).startswith("cancel"):
        _ok(f"ALPACA confirms cancelled: status={broker.get('status')}")
    else:
        status = broker.get("status") if broker else "<not found>"
        _warn(f"Alpaca cancel not confirmed within 10s (status={status})")

    rep = _wait_log_status(ex, order_id, {OrderStatus.CANCELLED}, timeout=3.0)
    if rep is not None:
        _ok("local model recorded CANCELLED via WS")
    else:
        _warn("local model did not record CANCELLED within 3s (WS lag)")
    if order_id not in [p.order_id for p in ex.get_pending_orders()]:
        _ok("cleared from local pending queue")


def _final_verification(ex: AlpacaPaperExecutor, probe: _AlpacaProbe) -> None:
    _hr("STEP 6 — reset() + independent final state check")
    t0 = time.perf_counter()
    ex.reset()
    _record_latency("reset RTT", time.perf_counter() - t0)
    _ok("reset() completed (account-wide flatten + local clear)")
    # Allow broker-side cancellations to settle before verification.
    time.sleep(1.0)
    positions = probe.get_positions()
    open_orders = probe.get_open_orders()
    print(f"  ALPACA positions after reset:  {len(positions)}")
    print(f"  ALPACA open orders after reset: {len(open_orders)}")
    if not positions:
        _ok("Alpaca account flat (no positions)")
    else:
        _warn(f"Alpaca still shows positions: {positions}")
    if not open_orders:
        _ok("Alpaca account has no open orders")
    else:
        _warn(f"Alpaca still shows open orders: {open_orders}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="SPY", help="ticker to trade")
    p.add_argument("--qty", type=float, default=1.0,
                   help="whole-share quantity (IOC/GTC need whole shares)")
    p.add_argument("--skip-fills", action="store_true",
                   help="skip market orders / fill waits (reads + limit/cancel only)")
    p.add_argument("--limit-offset-pct", type=float, default=0.10,
                   help="how far below market the resting limit sits (0.10 = 10%%)")
    p.add_argument("--fill-timeout", type=float, default=15.0,
                   help="seconds to wait for a market-order fill confirmation")
    p.add_argument("--ws-warmup", type=float, default=2.5,
                   help="seconds to let the trade_updates WS connect before "
                        "the first market order")
    p.add_argument("--no-flatten-on-start", action="store_true",
                   help="don't auto-flatten a dirty account (let __init__ raise)")
    args = p.parse_args()

    import os

    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    api_secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not api_key or not api_secret:
        _fail("ALPACA_API_KEY / ALPACA_SECRET_KEY missing in env / .env")
        return 2

    probe = _AlpacaProbe(api_key, api_secret)
    provider = AlpacaProvider()

    _hr("STEP 1 — construct AlpacaPaperExecutor (creds + clean-start check)")
    try:
        ex = AlpacaPaperExecutor(
            flatten_on_start=not args.no_flatten_on_start,
        )
    except Exception as exc:
        _fail(f"executor construction failed: {exc!r}")
        return 3
    _ok(f"executor constructed; cash seeded from /v2/account = {ex.get_cash()}")

    try:
        _check_reads(ex)

        if not args.skip_fills:
            print(f"\n  warming up trade_updates WS for {args.ws_warmup:.1f}s ...")
            time.sleep(args.ws_warmup)
            price = _reference_price(provider, args.symbol)
            _check_market_order(
                ex, probe,
                side=ActionVerb.BUY,
                symbol=args.symbol,
                qty=args.qty,
                price=price,
                fill_timeout=args.fill_timeout,
            )
        else:
            price = _reference_price(provider, args.symbol)
            _warn("--skip-fills set: no market orders will be submitted")

        _check_limit_and_cancel(
            ex, probe,
            symbol=args.symbol,
            qty=args.qty,
            price=price,
            offset_pct=args.limit_offset_pct,
        )

        if not args.skip_fills:
            # Flatten the position we opened with the market BUY. Sell only
            # what the broker actually holds (handles a partial fill).
            held = ex.get_positions().get(args.symbol, 0.0)
            sell_qty = round(held, 6)
            if sell_qty > 0:
                _check_market_order(
                    ex, probe,
                    side=ActionVerb.SELL,
                    symbol=args.symbol,
                    qty=sell_qty,
                    price=_reference_price(provider, args.symbol),
                    fill_timeout=args.fill_timeout,
                )
            else:
                _warn(
                    f"no {args.symbol} position to flatten "
                    f"(market BUY may not have filled)"
                )
    finally:
        # Always leave the paper account flat, even if a step raised.
        try:
            _final_verification(ex, probe)
        except Exception as exc:
            _fail(f"final verification/reset error: {exc!r}")
        ex.close()
        _ok("executor closed (daemons stopped)")

    _print_latency_summary()

    _hr("DONE")
    print(
        "Read each section above: '[OK] ALPACA confirms ...' lines are the "
        "authoritative proof the order reached Alpaca. '[WARN] local model "
        "...' lines about WS lag are non-fatal — they mean the executor's "
        "local mirror trailed the broker, not that Alpaca missed the order."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
