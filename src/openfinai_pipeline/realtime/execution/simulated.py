"""Simulated execution with cash accounting and a pending-order queue.

Long orders reserve cash; short exposure plus pending commitments is capped at
account equity. Scalar quotes fill at the current price, while OHLC quotes may
trigger intrabar. Market, limit, stop, and stop-limit orders support slippage,
fees, cancellation, and expiry. PnL includes filled trades only.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

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

logger = logging.getLogger(__name__)


def _new_order_id() -> str:
    return f"sim-{uuid.uuid4().hex[:12]}"


class SimulatedExecutor(BaseExecutor):
    """Simulated paper-trading executor with cash + pending queue.

    Args:
        config: Slippage / transaction-cost / mode settings.
            ``execution_mode`` is informational here.
        initial_capital: Starting cash. Default 100000.0. Buy submissions
            reserve cash ``quantity * reference_price``; if reserved cash
            exceeds free cash the submission is rejected.
    """

    def __init__(
        self,
        config: TradingConfig | None = None,
        *,
        initial_capital: float = 100000.0,
    ) -> None:
        self._config = config or TradingConfig()
        self._initial_capital = float(initial_capital)
        self._cash: float = self._initial_capital
        self._reserved_cash: float = 0.0
        self._positions: dict[str, float] = {}
        self._reserved_positions: dict[str, float] = {}
        self._pending: dict[str, PendingOrder] = {}
        self._trade_log: list[ExecutionReport] = []
        # Last-seen price per symbol — used only to value open positions for
        # the 1x buying-power cap (PnL/rewards still take explicit prices).
        self._last_price: dict[str, float] = {}

    # BaseExecutor interface — submission lifecycle

    def submit(
        self,
        intent: OrderIntent,
        market_price: "float | dict[str, Any]",
        *,
        timestamp: datetime | None = None,
        step: int = 0,
    ) -> SubmitResult:
        ts = timestamp or datetime.now(timezone.utc)
        # Quote may be a scalar tick (realtime) or an OHLC bar (offline).
        low, high, ref = self._quote_bounds(market_price)
        if ref > 0.0:
            self._last_price[intent.symbol] = ref

        # Hold is a no-op fill kept for trade-log consistency.
        if intent.action == ActionVerb.HOLD:
            report = ExecutionReport(
                action=ActionVerb.HOLD,
                symbol=intent.symbol,
                quantity=0.0,
                market_price=ref,
                executed_price=ref,
                timestamp=ts,
                order_id=_new_order_id(),
                order_type=OrderType.MARKET,
                tif=TimeInForce.IOC,
                status=OrderStatus.FILLED,
                submitted_step=step,
                filled_step=step,
                client_tag=intent.client_tag,
            )
            self._trade_log.append(report)
            return SubmitResult(kind="filled", fill=report)

        if intent.action not in (ActionVerb.BUY, ActionVerb.SELL):
            return SubmitResult(
                kind="rejected",
                rejection=Rejection(
                    order_intent=intent,
                    reason_code=RejectionCode.INVALID_ACTION,
                    reason=(
                        f"Unknown action: {intent.action!r} "
                        "(expected buy/sell/hold)"
                    ),
                ),
            )

        if not (intent.quantity > 0.0):
            return SubmitResult(
                kind="rejected",
                rejection=Rejection(
                    order_intent=intent,
                    reason_code=RejectionCode.INVALID_QUANTITY,
                    reason=(
                        f"Quantity must be positive for {intent.action}, "
                        f"got {intent.quantity}"
                    ),
                ),
            )

        # Reference price for cash reservation: market_price for market,
        # the trigger price for limit/stop (pessimistic — never reserve
        # less than the trade could cost).
        ref_price = self._reference_price_for_reservation(intent, ref)
        if ref_price is None or ref_price <= 0.0:
            return SubmitResult(
                kind="rejected",
                rejection=Rejection(
                    order_intent=intent,
                    reason_code=RejectionCode.INVALID_PRICE,
                    reason=(
                        "could not derive a positive reference price for "
                        f"reservation (quote ref={ref})"
                    ),
                ),
            )

        # Margin check on buys; position check on sells.
        if intent.action == ActionVerb.BUY:
            slippage = self._config.slippage_pct or 0.0
            tx_cost = self._config.transaction_cost_pct or 0.0
            # Reserve at worst-case slippage + tx cost so a marketable
            # fill at a worse price can't over-extend.
            unit_cost = ref_price * (1.0 + max(slippage, 0.0))
            unit_cost *= (1.0 + max(tx_cost, 0.0))
            reservation = unit_cost * intent.quantity
            free_cash = self._cash - self._reserved_cash
            if reservation > free_cash + 1e-9:
                return SubmitResult(
                    kind="rejected",
                    rejection=Rejection(
                        order_intent=intent,
                        reason_code=RejectionCode.INSUFFICIENT_CASH,
                        reason=(
                            f"buy {intent.symbol} qty={intent.quantity} "
                            f"needs ~{reservation:.4f} but only "
                            f"{free_cash:.4f} free cash "
                            f"(cash={self._cash:.4f}, "
                            f"reserved={self._reserved_cash:.4f})"
                        ),
                    ),
                )
            position_reservation = 0.0
            cash_reservation = reservation
        else:  # sell / short
            # Sells never reserve position. A sell that only reduces a long
            # (new_pos >= 0) lowers exposure and is always allowed. A sell
            # that opens/extends a short consumes buying power: the strict
            # 1x cap rejects it if the resulting gross exposure (filled book
            # + this short + pending commitments) would exceed equity.
            position_reservation = 0.0
            cash_reservation = 0.0
            current_pos = self._positions.get(intent.symbol, 0.0)
            new_pos = current_pos - intent.quantity
            if new_pos < 0.0:
                equity, projected_gross = self._equity_and_gross(
                    intent.symbol, new_pos
                )
                if projected_gross + self._reserved_cash > equity + 1e-9:
                    return SubmitResult(
                        kind="rejected",
                        rejection=Rejection(
                            order_intent=intent,
                            reason_code=RejectionCode.INSUFFICIENT_CASH,
                            reason=(
                                f"short {intent.symbol} qty={intent.quantity} "
                                f"would raise gross exposure to "
                                f"~{projected_gross + self._reserved_cash:.4f} "
                                f"above equity {equity:.4f} (strict 1x "
                                "buying-power cap)"
                            ),
                        ),
                    )
                # A queued short reserves buying power for the exposure it
                # *adds* (the net new short units), released on fill/cancel/
                # expire like a buy reservation. Market/marketable shorts fill
                # synchronously and never consume this.
                short_increase = abs(new_pos) - max(0.0, -current_pos)
                cash_reservation = max(0.0, short_increase) * ref_price

        # Market: synchronous fill, no reserve-then-release dance —
        # debits cash + positions in one shot. Reservations only apply
        # on the queue path below.
        if intent.order_type == OrderType.MARKET:
            report = self._fill_market(
                intent,
                market_price=ref,
                timestamp=ts,
                step=step,
            )
            return SubmitResult(kind="filled", fill=report)

        # Marketable limit: would fill at market under normal broker
        # behaviour, but per the pessimistic convention we fill at
        # limit_price exactly (no improvement). Matches what tick()
        # would do on the next tick — kept here for sync-call parity.
        if intent.order_type == OrderType.LIMIT and self._is_limit_marketable(
            intent, low, high
        ):
            report = self._fill_limit_at_limit_price(
                intent,
                market_price=ref,
                timestamp=ts,
                step=step,
                filled_step=step,
            )
            return SubmitResult(kind="filled", fill=report)

        # Marketable stop / stop_limit orders (already-triggered) have
        # the same property; they fill synchronously at stop_price (or
        # cascade to limit for stop_limit).
        if intent.order_type == OrderType.STOP and self._is_stop_triggered(
            intent, low, high
        ):
            report = self._fill_stop_at_stop_price(
                intent,
                market_price=ref,
                timestamp=ts,
                step=step,
                filled_step=step,
            )
            return SubmitResult(kind="filled", fill=report)

        if intent.order_type == OrderType.STOP_LIMIT and self._is_stop_triggered(
            intent, low, high
        ):
            # Stop fired immediately. Now check the limit leg.
            if self._is_limit_marketable(intent, low, high):
                report = self._fill_limit_at_limit_price(
                    intent,
                    market_price=ref,
                    timestamp=ts,
                    step=step,
                    filled_step=step,
                )
                return SubmitResult(kind="filled", fill=report)
            # Stop fired, limit not marketable: queue as a triggered
            # stop_limit (active limit) for tick() to finish.
            return self._queue_pending(
                intent,
                cash_reservation=cash_reservation,
                position_reservation=position_reservation,
                timestamp=ts,
                step=step,
                triggered=True,
            )

        # IOC = "this tick or never": didn't fill sync → expire.
        if intent.tif == TimeInForce.IOC:
            report = self._record_unfilled_expiration(
                intent,
                market_price=ref,
                timestamp=ts,
                step=step,
                reason_code="ioc_unfilled_on_submit",
            )
            # IOC non-fill surfaces as rejection so the agent learns
            # "this couldn't fill". No reservation was made, no release
            # needed.
            return SubmitResult(
                kind="rejected",
                rejection=Rejection(
                    order_intent=intent,
                    reason_code="ioc_unfilled",
                    reason=(
                        f"IOC {intent.order_type} {intent.action} "
                        f"{intent.symbol} qty={intent.quantity} did not "
                        "fill at submission and IOC does not queue"
                    ),
                ),
            )

        # GTC: queue with reservation.
        return self._queue_pending(
            intent,
            cash_reservation=cash_reservation,
            position_reservation=position_reservation,
            timestamp=ts,
            step=step,
            triggered=False,
        )

    def cancel(self, order_id: str) -> CancelResult:
        order = self._pending.pop(order_id, None)
        if order is None:
            return CancelResult(
                kind="rejected",
                rejection=Rejection(
                    order_intent={"action": "cancel", "order_id": order_id},
                    reason_code=RejectionCode.ORDER_NOT_FOUND,
                    reason=f"no pending order with order_id={order_id!r}",
                ),
            )
        self._release_reservation(order)
        # Record cancellation in the trade log so the audit shows it.
        report = ExecutionReport(
            action=order.action,
            symbol=order.symbol,
            quantity=order.quantity,
            market_price=0.0,
            executed_price=0.0,
            timestamp=datetime.now(timezone.utc),
            order_id=order.order_id,
            order_type=order.order_type,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            tif=order.tif,
            status=OrderStatus.CANCELLED,
            submitted_step=order.submitted_step,
            filled_step=0,
            client_tag=order.client_tag,
        )
        self._trade_log.append(report)
        return CancelResult(kind="cancelled", cancelled=order)

    def tick(
        self,
        prices: "dict[str, float | dict[str, Any]]",
        *,
        step: int = 0,
        timestamp: datetime | None = None,
    ) -> TickResult:
        ts = timestamp or datetime.now(timezone.utc)
        result = TickResult()
        # Refresh the per-symbol price cache (used to value open positions
        # for the 1x buying-power cap) from every quote in this tick.
        for sym, quote in prices.items():
            ref = self._quote_bounds(quote)[2]
            if ref > 0.0:
                self._last_price[sym] = ref
        # Iterate over a snapshot because we mutate _pending inside.
        for order_id, order in list(self._pending.items()):
            quote = prices.get(order.symbol)
            if quote is None:
                # No price this tick: GTC stays pending; IOC expires
                # (we still can't check the trigger).
                if order.tif == TimeInForce.IOC:
                    self._pending.pop(order_id, None)
                    self._release_reservation(order)
                    expired = self._record_expiration(
                        order, market_price=0.0, timestamp=ts, step=step
                    )
                    result.expirations.append(expired)
                continue

            low, high, ref = self._quote_bounds(quote)

            # stop_limit: check stop first; fall through to limit leg
            # whether already triggered or just-triggered.
            if order.order_type == OrderType.STOP_LIMIT and not order.triggered:
                if self._is_stop_triggered_pending(order, low, high):
                    order.triggered = True

            filled_now = False
            executed_price: float | None = None

            if order.order_type == OrderType.LIMIT:
                if self._is_limit_marketable_pending(order, low, high):
                    executed_price = order.limit_price
                    filled_now = True
            elif order.order_type == OrderType.STOP:
                if self._is_stop_triggered_pending(order, low, high):
                    executed_price = order.stop_price
                    filled_now = True
            elif order.order_type == OrderType.STOP_LIMIT:
                if order.triggered and self._is_limit_marketable_pending(
                    order, low, high
                ):
                    executed_price = order.limit_price
                    filled_now = True
            else:
                # Market orders should never live in the pending queue;
                # log + drop defensively.
                logger.warning(
                    "tick(): unexpected market order in pending queue: %s",
                    order_id,
                )
                self._pending.pop(order_id, None)
                self._release_reservation(order)
                continue

            if filled_now:
                self._pending.pop(order_id, None)
                report = self._fill_pending(
                    order,
                    market_price=ref,
                    executed_price=executed_price or ref,
                    timestamp=ts,
                    step=step,
                )
                result.fills.append(report)
                continue

            # Not filled this tick. IOC: expire.
            if order.tif == TimeInForce.IOC:
                self._pending.pop(order_id, None)
                self._release_reservation(order)
                expired = self._record_expiration(
                    order, market_price=ref, timestamp=ts, step=step
                )
                result.expirations.append(expired)

        return result

    def expire_all(
        self,
        prices: dict[str, float] | None = None,
        *,
        step: int = 0,
        timestamp: datetime | None = None,
    ) -> list[ExecutionReport]:
        ts = timestamp or datetime.now(timezone.utc)
        prices = prices or {}
        out: list[ExecutionReport] = []
        for order_id, order in list(self._pending.items()):
            self._pending.pop(order_id, None)
            self._release_reservation(order)
            mp = prices.get(order.symbol, 0.0)
            expired = self._record_expiration(
                order, market_price=mp, timestamp=ts, step=step
            )
            out.append(expired)
        return out

    # BaseExecutor interface — accessors

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def get_pending_orders(self) -> list[PendingOrder]:
        return list(self._pending.values())

    def get_trade_log(self) -> list[ExecutionReport]:
        return list(self._trade_log)

    def get_cash(self) -> float | None:
        return self._cash

    def get_reserved_cash(self) -> float:
        return self._reserved_cash

    def compute_pnl(self, current_prices: dict[str, float]) -> dict[str, float]:
        """Compute PnL from the *filled* portion of the trade log."""
        pnl: dict[str, float] = defaultdict(float)
        for trade in self._trade_log:
            if trade.status != OrderStatus.FILLED:
                continue  # cancelled / expired / rejected don't count
            if trade.action == ActionVerb.HOLD:
                continue
            cur = current_prices.get(trade.symbol, trade.executed_price)
            if trade.action == ActionVerb.BUY:
                trade_pnl = trade.quantity * (cur - trade.executed_price)
            else:  # sell
                trade_pnl = trade.quantity * (trade.executed_price - cur)
            trade_pnl -= trade.transaction_cost
            pnl[trade.symbol] += trade_pnl

        total = sum(pnl.values())
        result = dict(pnl)
        result["__total"] = total
        return result

    def reset(self) -> None:
        self._cash = self._initial_capital
        self._reserved_cash = 0.0
        self._positions.clear()
        self._reserved_positions.clear()
        self._pending.clear()
        self._trade_log.clear()
        self._last_price.clear()

    # Internal helpers

    def _equity_and_gross(
        self, symbol: str, new_position: float
    ) -> tuple[float, float]:
        """Return ``(equity, projected_gross)`` for the 1x buying-power cap.

        ``equity`` is the net liquidation value of the current book
        (``cash + Σ position·price``); ``projected_gross`` is the absolute
        notional ``Σ |position|·price`` with ``symbol`` set to
        ``new_position``. Positions are valued at the last-seen quote per
        symbol; ``symbol``'s own price was cached from this order's quote.
        """
        equity = self._cash
        gross = 0.0
        seen = False
        for sym, pos in self._positions.items():
            price = self._last_price.get(sym, 0.0)
            equity += pos * price
            proj = new_position if sym == symbol else pos
            gross += abs(proj) * price
            if sym == symbol:
                seen = True
        if not seen:
            gross += abs(new_position) * self._last_price.get(symbol, 0.0)
        return equity, gross

    def _reference_price_for_reservation(
        self, intent: OrderIntent, market_price: float
    ) -> float | None:
        """Reservation reference price aligned with the pessimistic
        fill convention: each order type fills at the corresponding
        trigger price exactly, so the reservation matches the worst
        cash outflow we'll see at fill time. Slippage / tx multipliers
        are applied separately by the caller.
        """
        if intent.order_type == OrderType.MARKET:
            return market_price
        if intent.order_type == OrderType.LIMIT:
            # Pessimistic fill price for limits = limit_price.
            return intent.limit_price
        if intent.order_type == OrderType.STOP:
            return intent.stop_price
        if intent.order_type == OrderType.STOP_LIMIT:
            # Pessimistic: fills at limit_price after the stop fires.
            return intent.limit_price
        return None

    @staticmethod
    def _quote_bounds(quote: "float | dict[str, Any]") -> tuple[float, float, float]:
        """Normalise a quote to ``(low, high, ref)``.

        Scalar tick (realtime) -> ``(p, p, p)``. OHLC bar dict (offline
        replay) -> ``(low, high, close)``. Trigger checks use
        ``(low, high)`` (intrabar fills); fills + reservations use
        ``ref``. The scalar case collapses to ``low == high == ref`` so
        the realtime path is bit-identical to the pre-bar behaviour.
        """
        if isinstance(quote, dict):
            return (
                float(quote["low"]),
                float(quote["high"]),
                float(quote["close"]),
            )
        p = float(quote)
        return p, p, p

    def _is_limit_marketable(
        self, intent: OrderIntent, low: float, high: float
    ) -> bool:
        if intent.limit_price is None:
            return False
        if intent.action == ActionVerb.BUY:
            return low <= intent.limit_price
        return high >= intent.limit_price

    def _is_limit_marketable_pending(
        self, order: PendingOrder, low: float, high: float
    ) -> bool:
        if order.limit_price is None:
            return False
        if order.action == ActionVerb.BUY:
            return low <= order.limit_price
        return high >= order.limit_price

    def _is_stop_triggered(
        self, intent: OrderIntent, low: float, high: float
    ) -> bool:
        if intent.stop_price is None:
            return False
        if intent.action == ActionVerb.BUY:
            return high >= intent.stop_price
        return low <= intent.stop_price

    def _is_stop_triggered_pending(
        self, order: PendingOrder, low: float, high: float
    ) -> bool:
        if order.stop_price is None:
            return False
        if order.action == ActionVerb.BUY:
            return high >= order.stop_price
        return low <= order.stop_price

    def _fill_market(
        self,
        intent: OrderIntent,
        *,
        market_price: float,
        timestamp: datetime,
        step: int,
    ) -> ExecutionReport:
        slippage = self._config.slippage_pct or 0.0
        if intent.action == ActionVerb.BUY:
            executed_price = market_price * (1.0 + slippage)
        else:
            executed_price = market_price * (1.0 - slippage)
        return self._book_fill(
            intent,
            market_price=market_price,
            executed_price=executed_price,
            timestamp=timestamp,
            step=step,
            order_id=_new_order_id(),
            cash_reservation=0.0,
            position_reservation=0.0,
            filled_step=step,
            order_type=OrderType.MARKET,
            tif=TimeInForce.IOC,
        )

    def _fill_limit_at_limit_price(
        self,
        intent: OrderIntent,
        *,
        market_price: float,
        timestamp: datetime,
        step: int,
        filled_step: int,
    ) -> ExecutionReport:
        # Pessimistic: fill at limit_price exactly, no improvement.
        # |executed - market| is still reported as nominal slippage_cost
        # for audit even though limit semantics already cap the price.
        assert intent.limit_price is not None
        executed_price = float(intent.limit_price)
        return self._book_fill(
            intent,
            market_price=market_price,
            executed_price=executed_price,
            timestamp=timestamp,
            step=step,
            order_id=_new_order_id(),
            cash_reservation=0.0,
            position_reservation=0.0,
            filled_step=filled_step,
            order_type=intent.order_type,
            tif=intent.tif,
        )

    def _fill_stop_at_stop_price(
        self,
        intent: OrderIntent,
        *,
        market_price: float,
        timestamp: datetime,
        step: int,
        filled_step: int,
    ) -> ExecutionReport:
        assert intent.stop_price is not None
        executed_price = float(intent.stop_price)
        return self._book_fill(
            intent,
            market_price=market_price,
            executed_price=executed_price,
            timestamp=timestamp,
            step=step,
            order_id=_new_order_id(),
            cash_reservation=0.0,
            position_reservation=0.0,
            filled_step=filled_step,
            order_type=intent.order_type,
            tif=intent.tif,
        )

    def _fill_pending(
        self,
        order: PendingOrder,
        *,
        market_price: float,
        executed_price: float,
        timestamp: datetime,
        step: int,
    ) -> ExecutionReport:
        # Free reservation BEFORE booking the fill — _book_fill debits
        # actual cost from the unreserved pool. Reservation invariants
        # (worst-case ref_price at queue time) guarantee post-release
        # buckets always cover the fill.
        self._release_reservation(order)
        intent = OrderIntent(
            action=order.action,
            symbol=order.symbol,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            tif=order.tif,
            client_tag=order.client_tag,
        )
        return self._book_fill(
            intent,
            market_price=market_price,
            executed_price=executed_price,
            timestamp=timestamp,
            step=step,
            order_id=order.order_id,
            cash_reservation=0.0,        # already released above
            position_reservation=0.0,
            filled_step=step,
            order_type=order.order_type,
            tif=order.tif,
            submitted_step=order.submitted_step,
        )

    def _book_fill(
        self,
        intent: OrderIntent,
        *,
        market_price: float,
        executed_price: float,
        timestamp: datetime,
        step: int,
        order_id: str,
        cash_reservation: float,
        position_reservation: float,
        filled_step: int,
        order_type: str,
        tif: str,
        submitted_step: int | None = None,
    ) -> ExecutionReport:
        notional = executed_price * intent.quantity
        transaction_cost = notional * (self._config.transaction_cost_pct or 0.0)
        slippage_cost = abs(executed_price - market_price) * intent.quantity

        # Apply to cash + positions.
        if intent.action == ActionVerb.BUY:
            # Reservation was an estimate; release it then debit actual.
            self._reserved_cash -= cash_reservation
            if self._reserved_cash < 0:
                self._reserved_cash = 0.0
            self._cash -= notional + transaction_cost
            # Symmetric with the sell branch: a buy-to-cover that lands
            # flat (now reachable since shorting is allowed) drops the
            # symbol rather than leaving a 0.0 entry.
            new_pos = self._positions.get(intent.symbol, 0.0) + intent.quantity
            if abs(new_pos) < 1e-12:
                self._positions.pop(intent.symbol, None)
            else:
                self._positions[intent.symbol] = new_pos
        else:  # sell
            already_reserved = self._reserved_positions.get(intent.symbol, 0.0)
            self._reserved_positions[intent.symbol] = max(
                0.0, already_reserved - position_reservation
            )
            current_pos = self._positions.get(intent.symbol, 0.0)
            new_pos = current_pos - intent.quantity
            if abs(new_pos) < 1e-12:
                self._positions.pop(intent.symbol, None)
            else:
                self._positions[intent.symbol] = new_pos
            self._cash += notional - transaction_cost

        report = ExecutionReport(
            action=intent.action,
            symbol=intent.symbol,
            quantity=intent.quantity,
            market_price=market_price,
            executed_price=executed_price,
            slippage_cost=slippage_cost,
            transaction_cost=transaction_cost,
            timestamp=timestamp,
            order_id=order_id,
            order_type=order_type,
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            tif=tif,
            status=OrderStatus.FILLED,
            submitted_step=submitted_step if submitted_step is not None else step,
            filled_step=filled_step,
            client_tag=intent.client_tag,
        )
        self._trade_log.append(report)
        return report

    def _queue_pending(
        self,
        intent: OrderIntent,
        *,
        cash_reservation: float,
        position_reservation: float,
        timestamp: datetime,
        step: int,
        triggered: bool,
    ) -> SubmitResult:
        order_id = _new_order_id()
        order = PendingOrder(
            order_id=order_id,
            symbol=intent.symbol,
            action=intent.action,
            quantity=intent.quantity,
            order_type=intent.order_type,
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            tif=intent.tif,
            submitted_at=timestamp,
            submitted_step=step,
            reserved_cash=cash_reservation,
            reserved_position=position_reservation,
            triggered=triggered,
            client_tag=intent.client_tag,
        )
        self._pending[order_id] = order
        self._reserved_cash += cash_reservation
        if position_reservation > 0:
            self._reserved_positions[intent.symbol] = (
                self._reserved_positions.get(intent.symbol, 0.0)
                + position_reservation
            )
        return SubmitResult(kind="accepted", accepted=order)

    def _release_reservation(self, order: PendingOrder) -> None:
        if order.reserved_cash > 0:
            self._reserved_cash -= order.reserved_cash
            if self._reserved_cash < 0:
                self._reserved_cash = 0.0
            order.reserved_cash = 0.0
        if order.reserved_position > 0:
            current = self._reserved_positions.get(order.symbol, 0.0)
            new = max(0.0, current - order.reserved_position)
            if new == 0:
                self._reserved_positions.pop(order.symbol, None)
            else:
                self._reserved_positions[order.symbol] = new
            order.reserved_position = 0.0

    def _record_expiration(
        self,
        order: PendingOrder,
        *,
        market_price: float,
        timestamp: datetime,
        step: int,
    ) -> ExecutionReport:
        report = ExecutionReport(
            action=order.action,
            symbol=order.symbol,
            quantity=order.quantity,
            market_price=market_price,
            executed_price=0.0,
            timestamp=timestamp,
            order_id=order.order_id,
            order_type=order.order_type,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            tif=order.tif,
            status=OrderStatus.EXPIRED,
            submitted_step=order.submitted_step,
            filled_step=step,
            client_tag=order.client_tag,
        )
        self._trade_log.append(report)
        return report

    def _record_unfilled_expiration(
        self,
        intent: OrderIntent,
        *,
        market_price: float,
        timestamp: datetime,
        step: int,
        reason_code: str,
    ) -> ExecutionReport:
        report = ExecutionReport(
            action=intent.action,
            symbol=intent.symbol,
            quantity=intent.quantity,
            market_price=market_price,
            executed_price=0.0,
            timestamp=timestamp,
            order_id=_new_order_id(),
            order_type=intent.order_type,
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            tif=intent.tif,
            status=OrderStatus.EXPIRED,
            submitted_step=step,
            filled_step=step,
            client_tag=intent.client_tag,
            extra={"expire_reason": reason_code},
        )
        self._trade_log.append(report)
        return report
