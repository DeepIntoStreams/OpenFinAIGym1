"""Shared base for the offline (historical-replay) trading tasks.

``OfflineCryptoTrading`` and ``OfflineStockTrading`` are the same task fed
by different providers, so all of the replay machinery lives here and the
concrete tasks are thin shells that set the provider, cache directory,
default symbols, and metadata strings.

This is the historical-replay implementation of the data-source seam
defined on :class:`~openfinai_pipeline.benchmark.contracts.TradingTask`:
the engine, reward, observation skeleton, and evaluation are all inherited
from the base; only the per-symbol *market* observation block, the cursor
over cached OHLCV, and the OHLC-bar execution quotes are specialised here.
Order book is always ``None`` — historical bars carry no live book.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from openfinai_pipeline.benchmark.contracts import TaskMetadata, TradingTask
from openfinai_pipeline.realtime.data_providers.base import MarketSnapshot
from openfinai_pipeline.realtime.execution import OrderType
from openfinai_pipeline.realtime.execution.simulated import SimulatedExecutor
from openfinai_pipeline.realtime.tasks.base_realtime_task import (
    _resolve_context_resolutions,
    _validate_target_symbols,
)
from openfinai_pipeline.settings import TradingConfig

# datasets/ root, resolved relative to this file:
# benchmark/ -> openfinai_pipeline/ -> src/ -> <repo root>.
_DATASETS_ROOT = (
    Path(__file__).resolve().parents[3] / "data" / "pipeline_output" / "datasets"
)

# Default observation context depth (bars of recent history per step).
# 20 matches the longest hardcoded slice key (``returns_20h``).
_DEFAULT_CONTEXT_BARS = 20


def _parse_iso_date(value: str) -> datetime:
    if "T" in value:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _ts_to_epoch(timestamps: Any) -> np.ndarray:
    """Convert ISO-8601 timestamps to UTC epoch seconds.

    Used by the extra-resolution slicer to align lower-frequency bars with
    the current primary-frame step via ``numpy.searchsorted``. Missing /
    invalid entries surface as ``0`` so they sort before any real timestamp.
    """
    if timestamps is None:
        return np.zeros(0, dtype=np.int64)
    series = pd.Series(timestamps)
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    epochs = (parsed.astype("int64") // 10**9).to_numpy().copy()
    epochs[pd.isna(parsed)] = 0
    return epochs.astype(np.int64)


def _bars_to_df(bars: List[MarketSnapshot]) -> pd.DataFrame:
    rows = []
    for b in bars:
        rows.append(
            {
                "timestamp": b.timestamp.isoformat(),
                "open": float(b.open if b.open is not None else b.price),
                "high": float(b.high if b.high is not None else b.price),
                "low": float(b.low if b.low is not None else b.price),
                "close": float(b.close if b.close is not None else b.price),
                "volume": float(b.volume if b.volume is not None else 0.0),
            }
        )
    return pd.DataFrame(rows)


class _OfflineTradingTask(TradingTask):
    """Replay transactional trading over cached OHLCV.

    Observations contain causal market history and portfolio state. Actions may
    hold, cancel, or submit one or more market/conditional orders. Configuration
    controls data windows, capital, episode length, costs, and scored symbols.
    """

    # Subclass seams
    _CACHE_SUBDIR: str = ""            # datasets/ subdir for the CSV cache
    _DEFAULT_SYMBOLS: tuple[str, ...] = ()
    _SOURCE_LABEL: str = ""            # "Binance" / "Alpaca"
    _ASSET_TAG: str = ""               # "crypto" / "stock"
    _TASK_ID_PREFIX: str = ""          # "offline_crypto_trading" / ...
    _TITLE: str = ""                   # "Offline Crypto Trading" / ...
    _VERSION: str = "1.0.0"

    def _make_default_provider(self) -> Any:
        raise NotImplementedError

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        provider: Optional[Any] = None,
    ) -> None:
        super().__init__(config)
        self._symbols: List[str] = list(
            self.config.get("symbols", list(self._DEFAULT_SYMBOLS))
        )
        if not self._symbols:
            raise ValueError(f"{type(self).__name__} requires at least one symbol")
        context_resolutions = self.config.get(
            "context_resolutions",
            [{"interval": "1h", "bars": _DEFAULT_CONTEXT_BARS}],
        )
        (
            data_resolution_interval,
            data_resolution_bars,
            extras,
            _primary_lookback,
        ) = _resolve_context_resolutions(
            context_resolutions, self.config.get("data_resolution")
        )
        self._interval: str = data_resolution_interval
        self._context_bars: int = data_resolution_bars
        self._extra_resolutions: tuple[tuple[str, int], ...] = extras
        self._start: str = str(self.config.get("start", "2020-01-01"))
        self._end: str = str(self.config.get("end", "2023-01-01"))
        self._episode_length: int = int(self.config.get("episode_length", 500))
        self._start_offset: int = int(self.config.get("start_offset", 0))
        self._slippage_pct: float = float(self.config.get("slippage_pct", 0.0))
        self._transaction_cost_pct: float = float(
            self.config.get("transaction_cost_pct", 0.0)
        )
        self._target_symbols: List[str] = _validate_target_symbols(
            self.config.get("target_symbols"), self._symbols
        )
        self._provider = provider
        self._ohlcv: Dict[str, pd.DataFrame] = {}
        self._prices: Dict[str, np.ndarray] = {}
        self._ohlcv_extra: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._extra_ts: Dict[str, Dict[str, np.ndarray]] = {}
        self._primary_ts: Dict[str, np.ndarray] = {}
        self._min_len: int = 0
        self._executor = SimulatedExecutor(
            TradingConfig(
                slippage_pct=self._slippage_pct,
                transaction_cost_pct=self._transaction_cost_pct,
            ),
            initial_capital=float(self.config.get("initial_cash", 100000.0)),
        )

    def metadata(self) -> TaskMetadata:
        sym_label = "_".join(s.lower() for s in self._symbols)
        return TaskMetadata(
            task_id=f"{self._TASK_ID_PREFIX}_{sym_label}",
            title=f"{self._TITLE} ({', '.join(self._symbols)}, {self._interval} replay)",
            description=(
                f"Sequential trading on historical {self._SOURCE_LABEL} "
                f"{self._interval} bars for {', '.join(self._symbols)} from "
                f"{self._start} to {self._end}. The agent submits transactional "
                f"orders (market/limit/stop, fractional sizing, shorting) and "
                f"manages a buying-power-capped portfolio."
            ),
            interaction_model="trading",
            data_requirements=[
                f"{self._SOURCE_LABEL} {sym} {self._interval} OHLCV"
                for sym in self._symbols
            ],
            tags=[self._ASSET_TAG, "trading", self._interval],
            difficulty="medium",
            version=self._VERSION,
        )

    # Data loading

    @property
    def _cache_dir(self) -> Path:
        return _DATASETS_ROOT / self._CACHE_SUBDIR

    def _cache_path(self, symbol: str, interval: str | None = None) -> Path:
        ival = interval or self._interval
        return self._cache_dir / f"{symbol}_{ival}_{self._start}_{self._end}.csv"

    def _ensure_provider(self) -> Any:
        if self._provider is None:
            self._provider = self._make_default_provider()
        return self._provider

    def _load_symbol_ohlcv(
        self, symbol: str, interval: str | None = None
    ) -> pd.DataFrame:
        ival = interval or self._interval
        csv_path = self._cache_path(symbol, ival)
        if csv_path.exists():
            return pd.read_csv(csv_path)
        provider = self._ensure_provider()
        start_dt = _parse_iso_date(self._start)
        end_dt = _parse_iso_date(self._end)
        bars = provider.get_bars(symbol, ival, start_dt, end_dt)
        if not bars:
            raise ValueError(
                f"Provider returned no bars for {symbol} "
                f"[{self._start} .. {self._end}, {ival}]"
            )
        df = _bars_to_df(bars)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        return df

    def load_data(self) -> Any:
        if self._data is not None:
            return self._data

        per_symbol: Dict[str, pd.DataFrame] = {}
        per_symbol_prices: Dict[str, np.ndarray] = {}
        per_symbol_ts: Dict[str, np.ndarray] = {}
        min_len = None
        for symbol in self._symbols:
            df = self._load_symbol_ohlcv(symbol)
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)
            df["return_1h"] = df["close"].pct_change().fillna(0.0)
            df = df.reset_index(drop=True)
            per_symbol[symbol] = df
            per_symbol_prices[symbol] = df["close"].values.astype(np.float64)
            per_symbol_ts[symbol] = _ts_to_epoch(df.get("timestamp"))
            n = len(df)
            min_len = n if min_len is None else min(min_len, n)

        self._ohlcv = per_symbol
        self._prices = per_symbol_prices
        self._primary_ts = per_symbol_ts
        self._min_len = int(min_len or 0)
        self._data = per_symbol

        for ex_interval, _ex_bars in self._extra_resolutions:
            ex_per_sym: Dict[str, pd.DataFrame] = {}
            ex_per_sym_ts: Dict[str, np.ndarray] = {}
            for symbol in self._symbols:
                ex_df = self._load_symbol_ohlcv(symbol, ex_interval)
                for col in ("open", "high", "low", "close", "volume"):
                    ex_df[col] = ex_df[col].astype(float)
                ex_df = ex_df.reset_index(drop=True)
                ex_per_sym[symbol] = ex_df
                ex_per_sym_ts[symbol] = _ts_to_epoch(ex_df.get("timestamp"))
            self._ohlcv_extra[ex_interval] = ex_per_sym
            self._extra_ts[ex_interval] = ex_per_sym_ts
        return self._data

    # Spaces

    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "keys": {
                "step": "int",
                "steps_remaining": "int",
                "symbols": (
                    "dict[str -> {symbol, price, open, high, low, close, "
                    "volume, timestamp, return_1h, returns_5h, returns_20h, "
                    "recent_bars, order_book(None for historical replay)}]"
                ),
                "portfolio": (
                    "dict[cash, reserved_cash, positions, pnl, value, "
                    "pending_orders]"
                ),
            },
            "symbols": list(self._symbols),
        }

    def get_action_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "keys": {
                "action": "'buy' | 'sell' | 'hold' | 'cancel' (required)",
                "symbol": "str (required for buy/sell)",
                "quantity": "float (> 0 for buy/sell; ignored for hold/cancel)",
                "order_type": (
                    "'market' | 'limit' | 'stop' | 'stop_limit' (default 'market')"
                ),
                "limit_price": "float (required for limit / stop_limit)",
                "stop_price": "float (required for stop / stop_limit)",
                "tif": "'ioc' | 'gtc' (default 'gtc'; ignored for market/hold)",
                "order_id": "str (required only for action='cancel')",
            },
            "batch": (
                "{'orders': [<single-order dict>, ...]} for multiple orders / "
                "cancels in one step"
            ),
            "order_types": list(OrderType.ALL),
            "time_in_force": ["ioc", "gtc"],
            "symbols": list(self._symbols),
        }

    # Lifecycle

    def reset(self) -> Any:
        if self._data is None:
            self.load_data()
        self._executor.reset()
        return super().reset()

    # Data-source hooks (the historical-replay seam)

    def _current_idx(self) -> int:
        idx = self._start_offset + self._step_count
        return min(idx, self._min_len - 1)

    def _episode_end(self) -> int:
        max_bars = self._min_len - self._start_offset - 1
        if self._episode_length > 0:
            return min(self._episode_length, max_bars)
        return max_bars

    def _episode_done(self) -> bool:
        return self._step_count >= self._episode_end() - 1

    def _steps_remaining(self) -> int:
        return self._episode_end() - self._step_count

    def _current_prices(self) -> Dict[str, float]:
        idx = self._current_idx()
        return {sym: float(self._prices[sym][idx]) for sym in self._symbols}

    def _execution_quotes(self) -> Dict[str, Dict[str, float]]:
        idx = self._current_idx()
        quotes: Dict[str, Dict[str, float]] = {}
        for sym in self._symbols:
            row = self._ohlcv[sym].iloc[idx]
            quotes[sym] = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        return quotes

    def _market_observation_block(self) -> Dict[str, Dict[str, Any]]:
        idx = self._current_idx()
        sym_obs: Dict[str, Dict[str, Any]] = {}
        for sym in self._symbols:
            df = self._ohlcv[sym]
            row = df.iloc[idx]
            returns = df["return_1h"].values
            start = max(0, idx - self._context_bars)
            recent = returns[start : idx + 1].tolist()
            rb_start = max(0, idx - self._context_bars + 1)
            recent_bars = df.iloc[rb_start : idx + 1].to_dict("records")
            entry: Dict[str, Any] = {
                "symbol": sym,
                "price": float(row["close"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "timestamp": str(row["timestamp"]) if "timestamp" in row else None,
                "return_1h": float(row["return_1h"]),
                "returns_5h": recent[-5:] if len(recent) >= 5 else recent,
                "returns_20h": recent[-20:] if len(recent) >= 20 else recent,
                "recent_bars": recent_bars,
                # Historical bars carry no live order book.
                "order_book": None,
            }
            if self._extra_resolutions:
                entry["recent_bars_by_interval"] = self._slice_all_resolutions(
                    sym, idx
                )
            sym_obs[sym] = entry
        return sym_obs

    def _slice_all_resolutions(
        self, symbol: str, primary_idx: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Per-interval recent-bars mapping for ``symbol``.

        Includes the primary interval (sliced to ``context_bars`` rows
        ending at ``primary_idx``) plus every configured extra interval
        (sliced via ``np.searchsorted`` on epoch-second timestamps so the
        lower-frequency window ends at the same wall-clock cutoff as the
        primary). Each element is an OHLCV row dict.
        """
        out: Dict[str, List[Dict[str, Any]]] = {}
        df = self._ohlcv[symbol]
        start = max(0, primary_idx - self._context_bars + 1)
        out[self._interval] = df.iloc[start : primary_idx + 1].to_dict("records")
        primary_ts_arr = self._primary_ts.get(symbol)
        if primary_ts_arr is None or primary_idx >= len(primary_ts_arr):
            cutoff_ts: int = 0
        else:
            cutoff_ts = int(primary_ts_arr[primary_idx])
        for ex_interval, ex_bars in self._extra_resolutions:
            ex_df = self._ohlcv_extra.get(ex_interval, {}).get(symbol)
            ex_ts = self._extra_ts.get(ex_interval, {}).get(symbol)
            if ex_df is None or ex_ts is None or len(ex_ts) == 0 or cutoff_ts <= 0:
                out[ex_interval] = []
                continue
            cutoff = int(np.searchsorted(ex_ts, cutoff_ts, side="right")) - 1
            if cutoff < 0:
                out[ex_interval] = []
                continue
            ex_start = max(0, cutoff - ex_bars + 1)
            out[ex_interval] = ex_df.iloc[ex_start : cutoff + 1].to_dict("records")
        return out
