"""DataProvider protocol, MarketSnapshot data class, and interval helpers."""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class MarketSnapshot:
    """A single price observation for a symbol at a point in time."""

    symbol: str
    timestamp: datetime
    price: float
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderBookSnapshot:
    """Level-2 order book or NBBO snapshot.

    Providers that support full depth (e.g. Binance) populate ``bids``/``asks``
    with a list of ``(price, quantity)`` tuples.  Providers that only have
    NBBO (e.g. Alpaca) populate ``best_bid``/``best_ask`` and leave the lists
    empty.
    """

    symbol: str
    timestamp: datetime
    best_bid: float | None = None
    best_bid_qty: float | None = None
    best_ask: float | None = None
    best_ask_qty: float | None = None
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def spread(self) -> float | None:
        """Ask - bid spread, or ``None`` if either side is missing."""
        if self.best_ask is not None and self.best_bid is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> float | None:
        """Mid-point of best bid and ask, or ``None`` if either is missing."""
        if self.best_ask is not None and self.best_bid is not None:
            return (self.best_ask + self.best_bid) / 2.0
        return None


# Interval parsing

# Regex: one or more digits followed by a unit letter.
_INTERVAL_RE = re.compile(r"^(\d+)\s*([mhdwM]|min|hour|day|week|month)$", re.IGNORECASE)

# Canonical single-char unit for each recognised spelling.
_UNIT_ALIASES: dict[str, str] = {
    "m": "m",
    "min": "m",
    "h": "h",
    "hour": "h",
    "d": "d",
    "day": "d",
    "w": "w",
    "week": "w",
    "M": "M",
    "month": "M",
}


def parse_interval(interval: str) -> tuple[int, str]:
    """Parse a human-friendly interval string into ``(value, unit)``.

    Accepted inputs (case-insensitive except ``M`` for month):
        ``"1m"``, ``"30min"``, ``"2h"``, ``"4hour"``, ``"1d"``, ``"1day"``,
        ``"1w"``, ``"1week"``, ``"1M"``, ``"1month"``

    Returns e.g. ``(30, "m")``, ``(4, "h")``, ``(1, "d")``.

    Raises :class:`ValueError` if the string cannot be parsed.
    """
    match = _INTERVAL_RE.match(interval.strip())
    if not match:
        raise ValueError(
            f"Cannot parse interval {interval!r}.  "
            "Expected format like '1m', '30min', '4h', '1d', '1w', '1M'."
        )
    value = int(match.group(1))
    raw_unit = match.group(2)
    # Preserve case for "M" (month) vs "m" (minute).
    unit = _UNIT_ALIASES.get(raw_unit if raw_unit == "M" else raw_unit.lower())
    if unit is None:
        raise ValueError(f"Unknown interval unit {raw_unit!r} in {interval!r}")
    return value, unit


def to_binance_interval(interval: str) -> str:
    """Convert a generic interval string to the Binance kline format.

    Binance format: ``{n}{unit}`` where unit is one of
    ``m``, ``h``, ``d``, ``w``, ``M``.
    """
    value, unit = parse_interval(interval)
    return f"{value}{unit}"


def interval_to_timedelta(interval: str, n_bars: int = 1) -> timedelta:
    """Convert an interval string (× bar count) to a :class:`timedelta`.

    Useful for sizing backfill windows in bar-count terms regardless of
    the chosen interval — e.g. ``interval_to_timedelta("5m", 60)`` →
    300-minute window. Months are approximated as 30 days.

    Raises :class:`ValueError` for unparseable intervals.
    """
    value, unit = parse_interval(interval)
    total = value * max(int(n_bars), 0)
    if unit == "m":
        return timedelta(minutes=total)
    if unit == "h":
        return timedelta(hours=total)
    if unit == "d":
        return timedelta(days=total)
    if unit == "w":
        return timedelta(weeks=total)
    if unit == "M":
        return timedelta(days=30 * total)
    raise ValueError(f"Unsupported unit {unit!r} in interval {interval!r}")


def interval_to_seconds(interval: str) -> int:
    """Return the number of seconds spanned by one bar at *interval*.

    Used to compare interval coarseness (which is finer / coarser) and
    to align timestamps to clock-boundary buckets when downsampling.
    Months are approximated as 30 days, matching
    :func:`interval_to_timedelta`.
    """
    return int(interval_to_timedelta(interval, 1).total_seconds())


def resolve_at_for_horizon(
    submitted_at: datetime, interval: str, horizon_bars: int
) -> datetime:
    """Close timestamp of the bar ``horizon_bars`` ahead of submission.

    Realtime forecasts target an aligned bar boundary, mirroring offline
    forecasting's ``close.shift(-N)``: floor ``submitted_at`` onto the
    ``interval`` grid to find the bar the agent submitted within, advance
    ``horizon_bars`` whole bars, and return that target bar's **close**
    time (its open + one interval).

    The returned value is the "resolvable-after" instant stored as
    ``resolve_at``: the resolver gates on it (the target bar has closed by
    then) and fetches the target bar by stepping back one ``interval``.
    """
    seconds = interval_to_seconds(interval)
    if seconds <= 0:
        raise ValueError(f"interval {interval!r} spans non-positive seconds")
    horizon = max(int(horizon_bars), 1)
    submit_epoch = int(submitted_at.timestamp())
    submit_bar_open = (submit_epoch // seconds) * seconds
    target_bar_close = submit_bar_open + (horizon + 1) * seconds
    return datetime.fromtimestamp(target_bar_close, tz=timezone.utc)


def downsample_bars(
    primary_bars: list[MarketSnapshot],
    target_interval: str,
    target_bars: int,
) -> list[MarketSnapshot]:
    """Aggregate ``primary_bars`` into clock-aligned bars at ``target_interval``.

    Each output bar covers ``target_interval`` worth of input bars whose
    timestamps fall into the same clock-aligned bucket
    (``floor(epoch / target_seconds) * target_seconds``). OHLCV reduce:

    - ``open``  = first input bar's open (or price if no OHLC)
    - ``high``  = max over the bucket
    - ``low``   = min over the bucket
    - ``close`` = last input bar's close (or price)
    - ``volume`` = sum
    - ``timestamp`` = bucket start (the clock-aligned boundary)

    Returns the **last** ``target_bars`` aggregated buckets. The most
    recent bucket may be a partial (in-progress) target bar if
    ``primary_bars`` ends mid-target-window.
    """
    if not primary_bars or target_bars <= 0:
        return []
    target_seconds = interval_to_seconds(target_interval)
    if target_seconds <= 0:
        return []
    buckets: dict[int, list[MarketSnapshot]] = {}
    for bar in primary_bars:
        epoch = int(bar.timestamp.timestamp())
        bucket = (epoch // target_seconds) * target_seconds
        buckets.setdefault(bucket, []).append(bar)
    aggregated: list[MarketSnapshot] = []
    for bucket_epoch in sorted(buckets):
        in_bucket = buckets[bucket_epoch]
        symbol = in_bucket[0].symbol

        def _o(b: MarketSnapshot) -> float:
            return float(b.open if b.open is not None else b.price)

        def _h(b: MarketSnapshot) -> float:
            return float(b.high if b.high is not None else b.price)

        def _l(b: MarketSnapshot) -> float:
            return float(b.low if b.low is not None else b.price)

        def _c(b: MarketSnapshot) -> float:
            return float(b.close if b.close is not None else b.price)

        def _v(b: MarketSnapshot) -> float:
            return float(b.volume if b.volume is not None else 0.0)

        ts = datetime.fromtimestamp(bucket_epoch, tz=timezone.utc)
        aggregated.append(
            MarketSnapshot(
                symbol=symbol,
                timestamp=ts,
                price=_c(in_bucket[-1]),
                open=_o(in_bucket[0]),
                high=max(_h(b) for b in in_bucket),
                low=min(_l(b) for b in in_bucket),
                close=_c(in_bucket[-1]),
                volume=sum(_v(b) for b in in_bucket),
            )
        )
    return aggregated[-target_bars:]


def to_alpaca_timeframe(interval: str) -> str:
    """Convert a generic interval string to the Alpaca timeframe format.

    Alpaca format: ``{n}{UnitName}`` where UnitName is one of
    ``Min``, ``Hour``, ``Day``, ``Week``, ``Month``.
    """
    _unit_names = {"m": "Min", "h": "Hour", "d": "Day", "w": "Week", "M": "Month"}
    value, unit = parse_interval(interval)
    name = _unit_names.get(unit)
    if name is None:
        raise ValueError(f"No Alpaca timeframe mapping for unit {unit!r}")
    return f"{value}{name}"


@runtime_checkable
class DataProvider(Protocol):
    """Minimal contract for market-data backends (Binance, Alpaca, ...)."""

    name: str

    def get_current_price(self, symbol: str) -> MarketSnapshot:
        """Return the latest price for *symbol*."""
        ...

    def get_price_at(
        self, symbol: str, at: datetime, interval: str = "1m"
    ) -> MarketSnapshot:
        """Return the *interval* bar at timestamp *at* (its close is the price)."""
        ...

    def get_bars(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[MarketSnapshot]:
        """Return OHLCV bars for *symbol* between *start* and *end*."""
        ...


@runtime_checkable
class EventDataProvider(DataProvider, Protocol):
    """Extended contract for event-resolution markets (e.g. Polymarket).

    Event-resolution markets differ from continuous-asset markets in
    three ways the resolver needs to know about:

    1. **Discovery is dynamic.** Active markets come and go (a Polymarket
       market resolves once and is never reused), so the universe is
       fetched at trial-setup time via :meth:`discover_active_markets`
       rather than read from a static ``symbols`` list in ``task.toml``.
    2. **Resolution returns a discrete outcome**, not a price. The
       resolver calls :meth:`get_event_outcome` instead of
       :meth:`get_price_at`. Returns ``None`` while pending, ``0.0`` /
       ``1.0`` for binary YES/NO, or ``0.5`` for ambiguous (e.g. UMA
       dispute fallback) — predictions on ``0.5`` outcomes should be
       dropped from scoring.
    3. **Per-market metadata is rich** (question text, resolution
       criteria, end-date, orderbook). :meth:`get_market_metadata`
       returns the discovery payload for any known symbol so the task
       can refresh observation fields without re-querying the catalog.
    """

    def discover_active_markets(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """Return active markets matching *filters*, each as a dict.

        Filter keys are provider-specific. Each returned dict carries at
        minimum: ``symbol`` (the provider's stable market identifier
        used as the ``symbol`` in subsequent calls), ``question``,
        ``resolution_at`` (timezone-aware datetime), and
        ``current_price`` (current YES probability in [0, 1]).
        """
        ...

    def get_event_outcome(self, symbol: str) -> float | None:
        """Return the resolved outcome for *symbol*, or ``None`` if pending.

        ``0.0`` = NO, ``1.0`` = YES, ``0.5`` = ambiguous (e.g. UMA
        dispute fallback). Callers should treat ``0.5`` as
        unscoreable and drop the matching prediction.
        """
        ...

    def get_market_metadata(self, symbol: str) -> dict[str, Any]:
        """Return the full discovery payload for *symbol* (cached if seen)."""
        ...


@runtime_checkable
class StreamingDataProvider(DataProvider, Protocol):
    """Extended contract for providers that support WebSocket streaming.

    Providers implementing this protocol can push real-time bar and
    order-book updates via callbacks, in addition to the REST-based
    methods inherited from :class:`DataProvider`.
    """

    def get_order_book(self, symbol: str) -> OrderBookSnapshot | None:
        """Return current order book snapshot (REST).  ``None`` if unsupported."""
        ...

    def supports_websocket(self) -> bool:
        """Whether this provider can stream data via WebSocket."""
        ...

    async def subscribe_bars(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[MarketSnapshot], None],
    ) -> None:
        """Subscribe to real-time bar updates via WebSocket."""
        ...

    async def subscribe_order_book(
        self,
        symbol: str,
        callback: Callable[[OrderBookSnapshot], None],
    ) -> None:
        """Subscribe to order-book updates via WebSocket."""
        ...
