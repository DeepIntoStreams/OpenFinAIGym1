"""Shared order-lifecycle types and executor interface.

Executors support market, limit, stop, and stop-limit orders with IOC or GTC
time in force. Trading-business failures return tagged submit/cancel results
rather than raising; tasks expose rejections to the agent. ``execute`` remains
a compatibility wrapper for synchronous market orders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# Strings round-trip through JSON and TOML without an enum dependency.


class OrderType:
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

    ALL = ("market", "limit", "stop", "stop_limit")


class TimeInForce:
    IOC = "ioc"
    GTC = "gtc"

    ALL = ("ioc", "gtc")


class OrderStatus:
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"   # agent-initiated cancel
    EXPIRED = "expired"        # IOC didn't fill / episode ended
    REJECTED = "rejected"      # never accepted into the queue
    # Alpaca orders are booked at the submission snapshot until a trade-update
    # event supplies the final fill price or rejection.
    PROVISIONAL = "provisional"


class ActionVerb:
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CANCEL = "cancel"

    ALL = ("buy", "sell", "hold", "cancel")


class RejectionCode:
    UNKNOWN_SYMBOL = "unknown_symbol"
    INVALID_ACTION = "invalid_action"
    INVALID_ORDER_TYPE = "invalid_order_type"
    INVALID_TIF = "invalid_tif"
    INVALID_QUANTITY = "invalid_quantity"
    INVALID_PRICE = "invalid_price"
    MISSING_FIELD = "missing_field"
    INSUFFICIENT_CASH = "insufficient_cash"
    INSUFFICIENT_POSITION = "insufficient_position"
    ORDER_NOT_FOUND = "order_not_found"
    SCHEMA_ERROR = "schema_error"


# Intent + records


@dataclass
class OrderIntent:
    """Pre-submission shape -- what the agent submits.

    Constructed by :meth:`from_dict` from a (validated-or-not) dict
    coming off the agent. The validation lives on the *executor* side
    (so the executor's view of cash/positions can be checked at the
    same time); :meth:`from_dict` only catches malformed structure
    (missing keys, non-numeric quantities, ...) and returns a
    :class:`Rejection` for those.
    """

    action: str          # "buy" | "sell" -- normalised after parsing
    symbol: str
    quantity: float
    order_type: str = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    tif: str = TimeInForce.GTC
    client_tag: str | None = None  # optional agent-assigned hint, echoed in fills

    @classmethod
    def from_dict(
        cls, d: dict[str, Any]
    ) -> "OrderIntent | Rejection":
        """Parse + light validate. Returns either an OrderIntent or a Rejection.

        Heavy validation (cash, position) is the executor's job; this
        method only catches schema-level problems.
        """
        if not isinstance(d, dict):
            return Rejection(
                order_intent=d,
                reason_code=RejectionCode.SCHEMA_ERROR,
                reason=f"Expected dict, got {type(d).__name__}",
            )

        raw_action = d.get("action")
        if raw_action is None:
            return Rejection(
                order_intent=d,
                reason_code=RejectionCode.MISSING_FIELD,
                reason="action is required",
            )

        # The cancel verb is handled separately in the executor flow --
        # it is not a "submit"-shaped intent. We still parse it here so
        # callers can use a single from_dict path; the executor branches
        # on action.
        if raw_action == ActionVerb.HOLD:
            # hold is a no-op — represent as a market intent with qty=0
            return cls(
                action=ActionVerb.HOLD,
                symbol=str(d.get("symbol", "")),
                quantity=0.0,
                order_type=OrderType.MARKET,
                tif=TimeInForce.IOC,
            )

        if raw_action not in (ActionVerb.BUY, ActionVerb.SELL):
            return Rejection(
                order_intent=d,
                reason_code=RejectionCode.INVALID_ACTION,
                reason=(
                    f"action must be one of {sorted(ActionVerb.ALL)}, "
                    f"got {raw_action!r}"
                ),
            )

        symbol = d.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            return Rejection(
                order_intent=d,
                reason_code=RejectionCode.MISSING_FIELD,
                reason="symbol is required and must be a non-empty string",
            )

        try:
            quantity = float(d.get("quantity", 0.0))
        except (TypeError, ValueError):
            return Rejection(
                order_intent=d,
                reason_code=RejectionCode.INVALID_QUANTITY,
                reason=f"quantity must be numeric, got {d.get('quantity')!r}",
            )
        if not (quantity > 0.0):
            return Rejection(
                order_intent=d,
                reason_code=RejectionCode.INVALID_QUANTITY,
                reason=f"quantity must be > 0, got {quantity}",
            )

        order_type = d.get("order_type", OrderType.MARKET)
        if order_type not in OrderType.ALL:
            return Rejection(
                order_intent=d,
                reason_code=RejectionCode.INVALID_ORDER_TYPE,
                reason=(
                    f"order_type must be one of {sorted(OrderType.ALL)}, "
                    f"got {order_type!r}"
                ),
            )

        tif = d.get("tif", TimeInForce.GTC)
        if tif not in TimeInForce.ALL:
            return Rejection(
                order_intent=d,
                reason_code=RejectionCode.INVALID_TIF,
                reason=(
                    f"tif must be one of {sorted(TimeInForce.ALL)}, "
                    f"got {tif!r}"
                ),
            )

        # Required price fields per order_type.
        limit_price: float | None = None
        stop_price: float | None = None
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            raw_lp = d.get("limit_price")
            if raw_lp is None:
                return Rejection(
                    order_intent=d,
                    reason_code=RejectionCode.MISSING_FIELD,
                    reason=f"{order_type} order requires limit_price",
                )
            try:
                limit_price = float(raw_lp)
            except (TypeError, ValueError):
                return Rejection(
                    order_intent=d,
                    reason_code=RejectionCode.INVALID_PRICE,
                    reason=f"limit_price must be numeric, got {raw_lp!r}",
                )
            if not (limit_price > 0.0):
                return Rejection(
                    order_intent=d,
                    reason_code=RejectionCode.INVALID_PRICE,
                    reason=f"limit_price must be > 0, got {limit_price}",
                )
        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            raw_sp = d.get("stop_price")
            if raw_sp is None:
                return Rejection(
                    order_intent=d,
                    reason_code=RejectionCode.MISSING_FIELD,
                    reason=f"{order_type} order requires stop_price",
                )
            try:
                stop_price = float(raw_sp)
            except (TypeError, ValueError):
                return Rejection(
                    order_intent=d,
                    reason_code=RejectionCode.INVALID_PRICE,
                    reason=f"stop_price must be numeric, got {raw_sp!r}",
                )
            if not (stop_price > 0.0):
                return Rejection(
                    order_intent=d,
                    reason_code=RejectionCode.INVALID_PRICE,
                    reason=f"stop_price must be > 0, got {stop_price}",
                )

        client_tag = d.get("client_tag")
        if client_tag is not None and not isinstance(client_tag, str):
            client_tag = str(client_tag)

        return cls(
            action=raw_action,
            symbol=symbol,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            tif=tif,
            client_tag=client_tag,
        )


@dataclass
class PendingOrder:
    """Order sitting in the queue waiting to fill (or be cancelled)."""

    order_id: str
    symbol: str
    action: str               # "buy" | "sell"
    quantity: float
    order_type: str           # OrderType.*
    limit_price: float | None
    stop_price: float | None
    tif: str                  # TimeInForce.*
    submitted_at: datetime | None = None
    submitted_step: int = 0
    reserved_cash: float = 0.0       # cash held for buy orders
    reserved_position: float = 0.0   # position units held for sell orders
    triggered: bool = False           # for stop_limit: stop has fired, now a limit
    client_tag: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "tif": self.tif,
            "submitted_step": self.submitted_step,
            "triggered": self.triggered,
            "client_tag": self.client_tag,
        }


@dataclass
class ExecutionReport:
    """Record of a single trade execution (fill, hold, or expiration).

    The original two-arg market-fill shape is preserved (see ``action``
    / ``symbol`` / ``quantity`` / ``executed_price``); the new
    order-type fields are optional and default to backward-compatible
    market values when callers don't provide them.
    """

    action: str  # "buy" | "sell" | "hold"
    symbol: str
    quantity: float
    market_price: float
    executed_price: float
    slippage_cost: float = 0.0
    transaction_cost: float = 0.0
    timestamp: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # New fields -- default to "this is a synchronous market fill" so
    # existing tests that build ExecutionReport without them keep
    # working unchanged.
    order_id: str | None = None
    order_type: str = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    tif: str = TimeInForce.IOC
    status: str = OrderStatus.FILLED
    submitted_step: int = 0
    filled_step: int = 0
    client_tag: str | None = None

    @property
    def total_cost(self) -> float:
        return self.slippage_cost + self.transaction_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "action": self.action,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "market_price": self.market_price,
            "executed_price": self.executed_price,
            "slippage_cost": self.slippage_cost,
            "transaction_cost": self.transaction_cost,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "tif": self.tif,
            "status": self.status,
            "submitted_step": self.submitted_step,
            "filled_step": self.filled_step,
            "client_tag": self.client_tag,
        }


@dataclass
class Rejection:
    """Result of a refused submit/cancel.

    ``order_intent`` is the original dict (or partial dict) from the
    agent, surfaced verbatim so the agent can correlate. Code is the
    machine-readable enum in :class:`RejectionCode`; reason is the
    human string.
    """

    order_intent: Any
    reason_code: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_intent": self.order_intent,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


@dataclass
class SubmitResult:
    """Outcome of :meth:`BaseExecutor.submit`.

    ``kind`` is one of ``"filled"`` (market or marketable; immediate
    fill), ``"accepted"`` (queued in pending), or ``"rejected"``.
    Exactly one of the payload fields is populated.
    """

    kind: str  # "filled" | "accepted" | "rejected"
    fill: ExecutionReport | None = None
    accepted: PendingOrder | None = None
    rejection: Rejection | None = None

    @property
    def is_filled(self) -> bool:
        return self.kind == "filled"

    @property
    def is_accepted(self) -> bool:
        return self.kind == "accepted"

    @property
    def is_rejected(self) -> bool:
        return self.kind == "rejected"


@dataclass
class CancelResult:
    """Outcome of :meth:`BaseExecutor.cancel`.

    ``kind`` is one of ``"cancelled"`` or ``"rejected"``.
    """

    kind: str  # "cancelled" | "rejected"
    cancelled: PendingOrder | None = None
    rejection: Rejection | None = None

    @property
    def is_cancelled(self) -> bool:
        return self.kind == "cancelled"

    @property
    def is_rejected(self) -> bool:
        return self.kind == "rejected"


@dataclass
class TickResult:
    """Outcome of :meth:`BaseExecutor.tick`.

    Bundles the (possibly empty) lists of orders that filled and orders
    that expired during this tick. Callers translate the entries into
    the ``info`` dict returned to the agent.
    """

    fills: list[ExecutionReport] = field(default_factory=list)
    expirations: list[ExecutionReport] = field(default_factory=list)


# Abstract base


class BaseExecutor(ABC):
    """Abstract execution backend for paper trading.

    Implementations track positions, a chronological trade log of
    :class:`ExecutionReport`, optional cash + reserved cash, and a
    pending-order queue. PnL is computed from the trade log against
    current market prices.

    Subclasses must implement at minimum :meth:`submit`, :meth:`cancel`,
    :meth:`tick`, :meth:`get_positions`, :meth:`get_trade_log`,
    :meth:`compute_pnl`, and :meth:`reset`. The legacy synchronous
    :meth:`execute` is provided as a default that builds a market
    intent and routes through ``submit`` so existing callers keep
    working without backend-specific overrides.
    """

    @abstractmethod
    def submit(
        self,
        intent: OrderIntent,
        market_price: "float | dict[str, Any]",
        *,
        timestamp: datetime | None = None,
        step: int = 0,
    ) -> SubmitResult:
        """Submit an order. Never raises for trading-business errors;
        always returns a :class:`SubmitResult` whose ``kind`` is one of
        ``filled``, ``accepted``, or ``rejected``.

        ``market_price`` is a quote: a scalar tick price (realtime) or an
        OHLC bar dict ``{open,high,low,close}`` (offline replay). Backends
        that only see scalars (e.g. ``AlpacaPaperExecutor``) ignore the
        bar shape.
        """

    @abstractmethod
    def cancel(self, order_id: str) -> CancelResult:
        """Cancel a pending order. Never raises for missing ids;
        returns a :class:`CancelResult`.
        """

    @abstractmethod
    def tick(
        self,
        prices: "dict[str, float | dict[str, Any]]",
        *,
        step: int = 0,
        timestamp: datetime | None = None,
    ) -> TickResult:
        """Evaluate pending-order triggers and IOC expirations against
        ``prices`` (per-symbol quote — scalar tick or OHLC bar dict).
        Fills and expirations are appended to the trade log and returned
        in the :class:`TickResult`.
        """

    @abstractmethod
    def expire_all(
        self,
        prices: dict[str, float] | None = None,
        *,
        step: int = 0,
        timestamp: datetime | None = None,
    ) -> list[ExecutionReport]:
        """Expire every remaining pending order (episode-end clean-up).

        Each expiration is appended to the trade log with
        ``status=OrderStatus.EXPIRED`` and returned as an
        :class:`ExecutionReport` so the caller can surface them in
        ``info["expired"]``.
        """

    @abstractmethod
    def get_positions(self) -> dict[str, float]:
        """Return current position per symbol (symbol -> net quantity)."""

    @abstractmethod
    def get_pending_orders(self) -> list[PendingOrder]:
        """Return the current pending-order queue (chronological)."""

    @abstractmethod
    def get_trade_log(self) -> list[ExecutionReport]:
        """Return chronological list of all executed/expired trades."""

    @abstractmethod
    def compute_pnl(self, current_prices: dict[str, float]) -> dict[str, float]:
        """Compute per-symbol unrealised PnL from the trade log.

        Returns ``{symbol: pnl_value, ..., "__total": total_pnl}``.
        Only :attr:`OrderStatus.FILLED` entries contribute; rejected
        and expired entries are pure audit.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear positions, trade log, pending queue, and reserved cash."""

    # Optional bookkeeping surfaced by the realtime/offline tasks.
    # Subclasses without a meaningful concept of cash (e.g. the original
    # SimulatedExecutor before this refactor) override these.

    def get_cash(self) -> float | None:
        """Return current free cash, or ``None`` if not tracked."""
        return None

    def get_reserved_cash(self) -> float:
        """Return cash earmarked by pending buy orders (0.0 if not tracked)."""
        return 0.0

    # Backwards-compatible synchronous market-fill API

    def execute(
        self,
        action: str,
        symbol: str,
        quantity: float,
        market_price: float,
        timestamp: datetime | None = None,
    ) -> ExecutionReport:
        """Legacy synchronous market-order API.

        Builds a market intent and routes through :meth:`submit`. If
        the submission is rejected this method *does* raise
        :class:`ValueError` with the rejection's reason -- the legacy
        contract used exceptions for bad inputs and several existing
        callers still rely on that behaviour. New callers should use
        :meth:`submit` directly and inspect the returned
        :class:`SubmitResult`.
        """
        if action == ActionVerb.HOLD:
            intent = OrderIntent(
                action=ActionVerb.HOLD,
                symbol=symbol,
                quantity=0.0,
                order_type=OrderType.MARKET,
                tif=TimeInForce.IOC,
            )
        else:
            intent = OrderIntent(
                action=action,
                symbol=symbol,
                quantity=float(quantity),
                order_type=OrderType.MARKET,
                tif=TimeInForce.IOC,
            )
        result = self.submit(intent, market_price, timestamp=timestamp)
        if result.is_filled:
            assert result.fill is not None
            return result.fill
        if result.is_rejected:
            assert result.rejection is not None
            raise ValueError(result.rejection.reason)
        # Market intents must never end up "accepted" (queued) -- that
        # would silently break legacy callers that expect a synchronous
        # fill. Subclasses that route market through queue-then-fill
        # must still return kind="filled" for the caller.
        raise RuntimeError(
            f"execute() got unexpected SubmitResult.kind={result.kind!r} "
            "for a market order; submit() should fill market orders "
            "synchronously"
        )
