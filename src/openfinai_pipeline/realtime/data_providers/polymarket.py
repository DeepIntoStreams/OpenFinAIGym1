"""Polymarket Gamma/CLOB provider using condition IDs and YES-side prices."""

from __future__ import annotations

import json as _json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from openfinai_pipeline.realtime.data_providers.base import (
    MarketSnapshot,
    interval_to_seconds,
)

logger = logging.getLogger(__name__)


# CLOB accepts named intervals or a minute-based fidelity.
_INTERVAL_TO_CLOB_ENUM: dict[str, str] = {
    "1h": "1h",
    "6h": "6h",
    "1d": "1d",
    "1w": "1w",
}

# Conservative shared throttle for Gamma and CLOB.
_DEFAULT_RATE_LIMIT_PER_MIN = 120

_MAX_BATCH_CONDITION_IDS = 50


class PolymarketProvider:
    """Read market metadata, YES-price history, and resolved outcomes."""

    name = "polymarket"

    def __init__(
        self,
        gamma_base: str = "https://gamma-api.polymarket.com",
        clob_base: str = "https://clob.polymarket.com",
        rate_limit_per_min: int = _DEFAULT_RATE_LIMIT_PER_MIN,
        session: requests.Session | None = None,
    ) -> None:
        self._gamma_base = gamma_base.rstrip("/")
        self._clob_base = clob_base.rstrip("/")
        self._session = session or requests.Session()
        self._min_interval = 60.0 / max(rate_limit_per_min, 1)
        self._last_request: float = 0.0
        self._market_cache: dict[str, dict[str, Any]] = {}
        self._clob_token_cache: dict[str, str] = {}

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.monotonic()

    def _get(self, base: str, path: str, params: dict[str, Any] | None = None) -> Any:
        self._throttle()
        url = f"{base}{path}"
        resp = self._session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _get_gamma(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._get(self._gamma_base, path, params)

    def _get_clob(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._get(self._clob_base, path, params)

    @staticmethod
    def _parse_string_array(value: Any) -> list[Any]:
        """Parse either a native list or Gamma's JSON-encoded list form."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = _json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, TypeError):
                pass
        return []

    @classmethod
    def _market_to_snapshot_payload(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize a Gamma record into the cached public payload."""
        outcomes = cls._parse_string_array(raw.get("outcomes"))
        outcome_prices_raw = cls._parse_string_array(raw.get("outcomePrices"))
        outcome_prices: list[float] = []
        for p in outcome_prices_raw:
            try:
                outcome_prices.append(float(p))
            except (ValueError, TypeError):
                outcome_prices.append(float("nan"))
        clob_token_ids = [str(t) for t in cls._parse_string_array(raw.get("clobTokenIds"))]

        # endDateIso is date-only; prefer endDate's full timestamp.
        end_value = raw.get("endDate") or raw.get("endDateIso")
        resolution_at: datetime | None = None
        if end_value:
            try:
                resolution_at = datetime.fromisoformat(str(end_value).replace("Z", "+00:00"))
                if resolution_at.tzinfo is None:
                    resolution_at = resolution_at.replace(tzinfo=timezone.utc)
            except ValueError:
                resolution_at = None

        current_price = outcome_prices[0] if outcome_prices else float("nan")

        return {
            "symbol": str(raw.get("conditionId") or ""),
            "question": str(raw.get("question") or ""),
            "description": str(raw.get("description") or ""),
            "resolution_at": resolution_at,
            "current_price": current_price,
            "outcomes": outcomes,
            "outcome_prices": outcome_prices,
            "clob_token_ids": clob_token_ids,
            "best_bid": _safe_float(raw.get("bestBid")),
            "best_ask": _safe_float(raw.get("bestAsk")),
            "spread": _safe_float(raw.get("spread")),
            "volume": _safe_float(raw.get("volumeNum")) or _safe_float(raw.get("volume")),
            "volume_24h": _safe_float(raw.get("volume24hr")),
            "liquidity": _safe_float(raw.get("liquidityNum")) or _safe_float(raw.get("liquidity")),
            "closed": bool(raw.get("closed", False)),
            "active": bool(raw.get("active", True)),
            "uma_resolution_status": raw.get("umaResolutionStatus"),
            "tags": _coerce_str_list(raw.get("tags")),
            "categories": _coerce_str_list(raw.get("categories")),
            "slug": str(raw.get("slug") or ""),
            "raw": raw,  # full record for diagnostics
        }

    def _cache_market(self, payload: dict[str, Any]) -> None:
        symbol = payload["symbol"]
        if not symbol:
            return
        self._market_cache[symbol] = payload
        if payload["clob_token_ids"]:
            self._clob_token_cache[symbol] = payload["clob_token_ids"][0]

    def _fetch_market_by_condition_id(self, symbol: str) -> dict[str, Any] | None:
        """Fetch and cache a single market by its conditionId."""
        try:
            data = self._get_gamma("/markets", {"condition_ids": symbol, "limit": 1})
        except requests.HTTPError as exc:
            logger.warning("Gamma fetch failed for %s: %s", symbol, exc)
            return None
        if not isinstance(data, list) or not data:
            return None
        payload = self._market_to_snapshot_payload(data[0])
        self._cache_market(payload)
        return payload

    def _get_yes_clob_token(self, symbol: str) -> str | None:
        token = self._clob_token_cache.get(symbol)
        if token is not None:
            return token
        cached = self._market_cache.get(symbol)
        if cached and cached["clob_token_ids"]:
            self._clob_token_cache[symbol] = cached["clob_token_ids"][0]
            return self._clob_token_cache[symbol]
        fetched = self._fetch_market_by_condition_id(symbol)
        if fetched and fetched["clob_token_ids"]:
            return fetched["clob_token_ids"][0]
        return None

    def get_current_price(self, symbol: str) -> MarketSnapshot:
        """Return the YES price with NBBO metadata in ``extra``."""
        payload = self._market_cache.get(symbol) or self._fetch_market_by_condition_id(symbol)
        if payload is None:
            raise ValueError(f"No polymarket market found for {symbol}")
        ts = datetime.now(timezone.utc)
        return MarketSnapshot(
            symbol=symbol,
            timestamp=ts,
            price=float(payload["current_price"]),
            extra={
                "best_bid": payload["best_bid"],
                "best_ask": payload["best_ask"],
                "spread": payload["spread"],
                "volume_24h": payload["volume_24h"],
                "liquidity": payload["liquidity"],
                "outcome_prices": payload["outcome_prices"],
            },
        )

    def get_price_at(
        self, symbol: str, at: datetime, interval: str = "1m"
    ) -> MarketSnapshot:
        """Return the nearest YES price within one hour of ``at``.

        ``interval`` exists for protocol compatibility and is unused.
        """
        token = self._get_yes_clob_token(symbol)
        if token is None:
            raise ValueError(
                f"No CLOB token id known for {symbol}; "
                "discover_active_markets or fetch the market first."
            )
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        start_ts = int((at - timedelta(hours=1)).timestamp())
        end_ts = int((at + timedelta(hours=1)).timestamp())
        data = self._get_clob(
            "/prices-history",
            {
                "market": token,
                "startTs": start_ts,
                "endTs": end_ts,
                "fidelity": 1,
            },
        )
        history = data.get("history") if isinstance(data, dict) else None
        if not history:
            raise ValueError(
                f"No CLOB price history for {symbol} around {at.isoformat()}"
            )
        target_ts = int(at.timestamp())
        nearest = min(history, key=lambda h: abs(int(h["t"]) - target_ts))
        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(int(nearest["t"]), tz=timezone.utc),
            price=float(nearest["p"]),
        )

    def get_bars(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[MarketSnapshot]:
        """Return historical YES-side price bars between *start* and *end*."""
        token = self._get_yes_clob_token(symbol)
        if token is None:
            raise ValueError(
                f"No CLOB token id known for {symbol}; "
                "discover_active_markets or fetch the market first."
            )
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        params: dict[str, Any] = {
            "market": token,
            "startTs": int(start.timestamp()),
            "endTs": int(end.timestamp()),
        }
        clob_enum = _INTERVAL_TO_CLOB_ENUM.get(interval.lower())
        if clob_enum:
            params["interval"] = clob_enum
        else:
            seconds = interval_to_seconds(interval)
            params["fidelity"] = max(1, seconds // 60)
        data = self._get_clob("/prices-history", params)
        history = data.get("history") if isinstance(data, dict) else None
        if not history:
            return []
        bars: list[MarketSnapshot] = []
        for entry in history:
            try:
                ts = datetime.fromtimestamp(int(entry["t"]), tz=timezone.utc)
                price = float(entry["p"])
            except (KeyError, ValueError, TypeError):
                continue
            bars.append(
                MarketSnapshot(
                    symbol=symbol,
                    timestamp=ts,
                    price=price,
                    close=price,
                )
            )
        return bars

    def discover_active_markets(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """Discover currently-active markets matching *filters*.

        Server filters are the resolution-window bounds, ``min_total_volume``,
        ``min_liquidity``, and ``tag_id``. Local filters cover disputes, YES
        price, 24-hour volume, depth, categories, and binary outcomes.
        ``max_markets_per_trial`` caps the cached result.
        """
        params: dict[str, Any] = {
            "closed": "false",
            "limit": 500,
        }
        now = datetime.now(timezone.utc)
        win_min_h = filters.get("resolution_window_hours_min")
        win_max_h = filters.get("resolution_window_hours_max")
        if win_min_h is not None:
            params["end_date_min"] = (
                now + timedelta(hours=float(win_min_h))
            ).isoformat()
        if win_max_h is not None:
            params["end_date_max"] = (
                now + timedelta(hours=float(win_max_h))
            ).isoformat()
        min_total_volume = filters.get("min_total_volume")
        if min_total_volume is not None:
            params["volume_num_min"] = float(min_total_volume)
        min_liquidity = filters.get("min_liquidity")
        if min_liquidity is not None:
            params["liquidity_num_min"] = float(min_liquidity)
        tag_id = filters.get("tag_id")
        if tag_id is not None:
            params["tag_id"] = int(tag_id)
        # Gamma status queries are inclusive; exclude disputes locally.

        try:
            raw_list = self._get_gamma("/markets", params)
        except requests.HTTPError as exc:
            logger.warning("Gamma /markets fetch failed: %s", exc)
            return []
        if not isinstance(raw_list, list):
            return []

        min_yes = float(filters.get("min_yes_price", 0.01))
        max_yes = float(filters.get("max_yes_price", 0.99))
        min_24h = filters.get("min_24h_volume_usd")
        min_24h_v = float(min_24h) if min_24h is not None else None
        min_depth = filters.get("min_orderbook_depth_usd")
        min_depth_v = float(min_depth) if min_depth is not None else None
        categories_filter = [str(c).lower() for c in (filters.get("categories") or [])]
        binary_only = bool(filters.get("binary_only", True))
        exclude_disputed = bool(filters.get("exclude_disputed", True))
        cap = int(filters.get("max_markets_per_trial", 50))

        # Unresolved disputes may retain closed=false after the event time.
        _SKEW = timedelta(seconds=60)
        win_min_dt: datetime | None = None
        win_max_dt: datetime | None = None
        if win_min_h is not None:
            win_min_dt = now + timedelta(hours=float(win_min_h)) - _SKEW
        if win_max_h is not None:
            win_max_dt = now + timedelta(hours=float(win_max_h)) + _SKEW

        kept: list[dict[str, Any]] = []
        for raw in raw_list:
            payload = self._market_to_snapshot_payload(raw)
            if payload["closed"]:
                continue
            if not payload["active"]:
                continue
            if exclude_disputed and (
                str(payload.get("uma_resolution_status") or "").lower() == "disputed"
            ):
                continue
            if binary_only and len(payload["outcomes"]) != 2:
                continue
            res_at = payload["resolution_at"]
            if res_at is None:
                continue
            if win_min_dt is not None and res_at < win_min_dt:
                continue
            if win_max_dt is not None and res_at > win_max_dt:
                continue
            cp = payload["current_price"]
            if cp != cp:  # NaN check
                continue
            if cp < min_yes or cp > max_yes:
                continue
            if min_24h_v is not None and (payload["volume_24h"] or 0.0) < min_24h_v:
                continue
            if min_depth_v is not None:
                # Without liquidity, use spread as the available depth proxy.
                liq = payload["liquidity"] or 0.0
                if liq and liq < min_depth_v:
                    continue
                if not liq:
                    spread = payload["spread"]
                    if spread is None or spread > 0.05:
                        continue
            if categories_filter:
                cats = [c.lower() for c in payload["categories"]]
                if not any(c in cats for c in categories_filter):
                    continue
            if not payload["symbol"]:
                continue
            if not payload["clob_token_ids"]:
                continue
            self._cache_market(payload)
            kept.append(payload)
            if len(kept) >= cap:
                break
        return kept

    def get_event_outcome(self, symbol: str) -> float | None:
        """Refetch and return 0, 0.5, or 1; return ``None`` while pending."""
        fresh = self._fetch_market_by_condition_id(symbol)
        if fresh is None:
            return None
        if not fresh["closed"]:
            return None
        outcome_prices = fresh["outcome_prices"]
        if not outcome_prices or len(outcome_prices) < 1:
            return None
        yes_price = outcome_prices[0]
        if yes_price != yes_price:  # NaN
            return None
        # Allow minor drift around the three contractual outcomes.
        if abs(yes_price - 1.0) < 1e-6:
            return 1.0
        if abs(yes_price - 0.0) < 1e-6:
            return 0.0
        if abs(yes_price - 0.5) < 1e-6:
            return 0.5
        # A noncanonical value is not safe to score yet.
        logger.warning(
            "Polymarket %s closed but outcomePrices=%s not in {0,0.5,1}; "
            "treating as pending",
            symbol,
            outcome_prices,
        )
        return None

    def get_market_metadata(self, symbol: str) -> dict[str, Any]:
        """Return cached market payload, fetching if not seen."""
        cached = self._market_cache.get(symbol)
        if cached is not None:
            return cached
        fetched = self._fetch_market_by_condition_id(symbol)
        if fetched is None:
            raise ValueError(f"No polymarket market found for {symbol}")
        return fetched

def _safe_float(value: Any) -> float | None:
    """Coerce to float; return None on parse failure or None input."""
    if value is None:
        return None
    try:
        f = float(value)
    except (ValueError, TypeError):
        return None
    if f != f:  # NaN
        return None
    return f


def _coerce_str_list(value: Any) -> list[str]:
    """Best-effort coerce to list[str]; tolerates None / dicts / strings."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = _json.loads(value)
            if isinstance(parsed, list):
                value = parsed
        except (ValueError, TypeError):
            return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                # Gamma tags may be structured records.
                for key in ("label", "slug", "name"):
                    if key in item and isinstance(item[key], str):
                        out.append(item[key])
                        break
        return out
    return []
