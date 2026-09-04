"""Curated task: Offline Crypto Forecasting (Binance hourly OHLCV).

Multi-symbol batch forecasting on Binance hourly bars. Data is fetched
on first use via ``BinanceProvider.get_bars()`` and cached as CSV under
``data/pipeline_output/datasets/binance_crypto_hourly_ohlcv/``. Subsequent
runs load directly from the cache without touching the network.

Interaction pattern (``batch_mode=True``)::

    features    = task.get_features()                   # {sym: (n_test, 10)}
    predictions = agent.act(features)                   # {sym: (n_test,)}
    rewards     = task.predict_and_evaluate(predictions) # per-symbol + aggregate

For the default single-symbol config the dict has one key. Multi-symbol
config (``{"symbols": ["BTCUSDT", "ETHUSDT"]}``) yields one entry per
symbol with the same feature schema.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from openfinai_pipeline.benchmark.contracts import ForecastingTask, TaskMetadata
from openfinai_pipeline.realtime.data_providers.base import MarketSnapshot
from openfinai_pipeline.realtime.data_providers.binance import BinanceProvider
from openfinai_pipeline.realtime.tasks.base_realtime_task import (
    _validate_target_symbols,
)
from openfinai_pipeline.utils.features import sanitize_engineered_features
from data.knowledge_base.rewards.reward_bank import (
    DirectionalAccuracy,
    MAELoss,
    MAPELoss,
    MSELoss,
    PearsonCorrelation,
    R2Score,
    RMSELoss,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO_ROOT / "data" / "pipeline_output" / "datasets" / "binance_crypto_hourly_ohlcv"

_ROLLING_WINDOWS = [5, 20]
# Largest backward dependency across features: rolling_mean_20h /
# rolling_std_20h /momentum_20h all need 20 prior bars. Used to size
# the expected head-NaN envelope when sanitizing engineered features.
_MOMENTUM_SHIFT = 20

#: Per-symbol metrics computed on the full (predicted_price, target_price)
#: series via the gt-at-init :class:`Loss` classes. The torch dispatch
#: keeps a single source of truth shared with the auto-pipeline assembled
#: evaluator — no re-implementation of MSE/MAE/MAPE in numpy here.
_PANEL_LOSS_CLASSES: tuple[tuple[str, type], ...] = (
    ("mse", MSELoss),
    ("rmse", RMSELoss),
    ("mae", MAELoss),
    ("mape", MAPELoss),
    ("r2", R2Score),
    ("pearson", PearsonCorrelation),
)

#: Macro-aggregated keys (mean across :attr:`_target_symbols`). Only
#: scale-free metrics are macro-averaged because cross-symbol prices
#: live on incompatible scales (BTCUSDT ≈ 60k, SPY ≈ 500); a raw-MSE
#: macro-mean would just track whichever symbol had the wider range.
_AGG_KEYS: tuple[str, ...] = ("mape", "r2", "pearson", "directional_accuracy")

#: Allowed values for ``headline_metric`` in
#: :class:`OfflineCryptoForecasting`. The headline must be a macro key
#: so cross-symbol comparability holds; per-symbol headlines (e.g.
#: ``mape_BTCUSDT``) are exposed in the score dict but not eligible.
_VALID_OFFLINE_HEADLINE: tuple[str, ...] = _AGG_KEYS

FEATURE_NAMES: list[str] = [
    "return_1h",
    "log_return_1h",
    "rolling_mean_5h",
    "rolling_std_5h",
    "rolling_mean_20h",
    "rolling_std_20h",
    "volume_change",
    "high_low_range",
    "close_open_range",
    "momentum_20h",
]


def _parse_iso_date(value: str) -> datetime:
    """Parse ``YYYY-MM-DD`` (or full ISO timestamp) into a UTC datetime."""
    if "T" in value:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_optional_split_date(value: Any) -> Optional[pd.Timestamp]:
    """Parse a user-supplied ``split_date`` config value.

    Accepts ``None``, ISO date strings, ISO datetime strings, or a
    pre-built :class:`pd.Timestamp` / :class:`datetime`. Naive inputs
    are localized to UTC (we keep all time-axis comparisons in UTC to
    match the upstream provider's timestamp encoding). Anything else
    raises :class:`ValueError` with a clear message — silently
    accepting a malformed split would put the cutoff in an unexpected
    place and the failure would only surface as wrong-sized test sets.
    """
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"split_date={value!r} is not a valid ISO timestamp"
        ) from exc
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _bars_to_df(bars: List[MarketSnapshot]) -> pd.DataFrame:
    """Convert ``list[MarketSnapshot]`` to a normalised OHLCV DataFrame."""
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


class OfflineCryptoForecasting(ForecastingTask):
    """Predict future absolute close prices from Binance OHLCV features.

    Configuration selects symbols, dates, horizon, split policy, headline
    metric, and aggregate target symbols. A shared cutoff keeps each symbol's
    test window calendar-aligned. ``provider`` may override the default
    :class:`BinanceProvider`.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        provider: Optional[Any] = None,
    ) -> None:
        super().__init__(config)
        self._symbols: List[str] = list(self.config.get("symbols", ["BTCUSDT"]))
        if not self._symbols:
            raise ValueError("OfflineCryptoForecasting requires at least one symbol")
        self._interval: str = str(self.config.get("interval", "1h"))
        self._start: str = str(self.config.get("start", "2020-01-01"))
        self._end: str = str(self.config.get("end", "2023-01-01"))
        self._train_ratio: float = float(self.config.get("train_ratio", 0.8))
        self._split_date: Optional[pd.Timestamp] = _parse_optional_split_date(
            self.config.get("split_date")
        )
        self._forecast_horizon_bars: int = int(
            self.config.get("forecast_horizon_bars", 1)
        )
        if self._forecast_horizon_bars < 1:
            raise ValueError(
                f"forecast_horizon_bars must be >= 1; got "
                f"{self._forecast_horizon_bars!r}"
            )
        self._headline_metric: str = str(
            self.config.get("headline_metric", "mape")
        )
        if self._headline_metric not in _VALID_OFFLINE_HEADLINE:
            raise ValueError(
                f"headline_metric={self._headline_metric!r} is not in "
                f"the offline forecasting macro panel. Valid options: "
                f"{sorted(_VALID_OFFLINE_HEADLINE)}"
            )
        self._target_symbols: List[str] = _validate_target_symbols(
            self.config.get("target_symbols"), self._symbols
        )
        self._provider = provider

    def metadata(self) -> TaskMetadata:
        sym_label = "_".join(s.lower() for s in self._symbols)
        return TaskMetadata(
            task_id=f"offline_crypto_forecasting_{sym_label}",
            title=f"Offline Crypto Forecasting ({', '.join(self._symbols)}, {self._interval})",
            description=(
                f"Predict the absolute close price "
                f"{self._forecast_horizon_bars} bar(s) ahead for "
                f"{', '.join(self._symbols)} from 10 OHLCV-derived "
                f"features. Binance {self._interval} bars from "
                f"{self._start} to {self._end}. Headline metric: "
                f"{self._headline_metric}."
            ),
            interaction_model="forecasting",
            data_requirements=[f"Binance {sym} {self._interval} OHLCV" for sym in self._symbols],
            tags=["crypto", "forecasting", self._interval],
            difficulty="medium",
            version="2.0.0",
        )

    # Data loading

    def _cache_path(self, symbol: str) -> Path:
        return _CACHE_DIR / f"{symbol}_{self._interval}_{self._start}_{self._end}.csv"

    def _ensure_provider(self) -> Any:
        if self._provider is None:
            self._provider = BinanceProvider()
        return self._provider

    def _load_symbol_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Load raw OHLCV for ``symbol``: from cache if present, else fetch + cache."""
        csv_path = self._cache_path(symbol)
        if csv_path.exists():
            return pd.read_csv(csv_path)
        provider = self._ensure_provider()
        start_dt = _parse_iso_date(self._start)
        end_dt = _parse_iso_date(self._end)
        bars = provider.get_bars(symbol, self._interval, start_dt, end_dt)
        if not bars:
            raise ValueError(
                f"Provider returned no bars for {symbol} "
                f"[{self._start} .. {self._end}, {self._interval}]"
            )
        df = _bars_to_df(bars)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        return df

    @staticmethod
    def _engineer_features(
        raw: pd.DataFrame,
        forecast_horizon_bars: int = 1,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute features + price-prediction target + reference price.

        Targets and references emitted per row:

        * ``target`` — ``close[t + forecast_horizon_bars]``: the agent's
          prediction objective (absolute future price).
        * ``reference`` — ``close[t]``: the current bar's close, used at
          scoring time to derive direction via
          ``sign(predicted_price - reference)``. Exposed in the agent-
          facing ``dataset.h5`` so agents can also recover relative
          changes if useful.

        Exchange-gap bars (volume == 0, OHLC frozen) propagate ``+inf``
        into ``volume_change`` on the *following* row. ``dropna()``
        alone doesn't catch ``inf`` — sklearn estimators then refuse
        the feature matrix, the agent's ``train.py`` crashes, and the
        verifier short-circuits to ``reward = 0`` with no metric panel.
        :func:`sanitize_engineered_features` converts ``inf → NaN``
        before the final ``dropna`` and logs a warning per anomalous
        row with the upstream raw bar so the data issue is visible
        host-side.
        """
        df = raw.copy()
        df["return_1h"] = df["close"].pct_change()
        df["log_return_1h"] = np.log(df["close"] / df["close"].shift(1))
        for w in _ROLLING_WINDOWS:
            df[f"rolling_mean_{w}h"] = df["return_1h"].rolling(w).mean()
            df[f"rolling_std_{w}h"] = df["return_1h"].rolling(w).std()
        df["volume_change"] = df["volume"].pct_change()
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
        df["close_open_range"] = (df["close"] - df["open"]) / df["open"]
        df["momentum_20h"] = df["close"] / df["close"].shift(_MOMENTUM_SHIFT) - 1.0
        df["reference"] = df["close"].astype(float)
        df["target"] = df["close"].shift(-forecast_horizon_bars).astype(float)

        cleaned, _ = sanitize_engineered_features(
            df,
            raw,
            FEATURE_NAMES,
            head_envelope=max(*_ROLLING_WINDOWS, _MOMENTUM_SHIFT),
            tail_envelope=forecast_horizon_bars,
            symbol=symbol,
        )
        return cleaned

    def load_data(self) -> Any:
        if self._data is not None:
            return self._data
        # First pass: feature-engineer each symbol, collect cleaned df +
        # its parsed UTC timestamps. Splitting is deferred so we can
        # apply one shared cutoff across all symbols.
        per_symbol_df: Dict[str, pd.DataFrame] = {}
        per_symbol_ts: Dict[str, pd.Series] = {}
        for symbol in self._symbols:
            raw = self._load_symbol_ohlcv(symbol)
            df = self._engineer_features(
                raw, self._forecast_horizon_bars, symbol=symbol
            )
            per_symbol_df[symbol] = df
            per_symbol_ts[symbol] = pd.to_datetime(
                df["timestamp"], utc=True
            ).reset_index(drop=True)
        cutoff = self._resolve_cutoff(per_symbol_ts)
        per_symbol: Dict[str, Dict[str, Any]] = {}
        for symbol, df in per_symbol_df.items():
            ts = per_symbol_ts[symbol]
            split = int((ts <= cutoff).sum())
            n_total = len(df)
            if split <= 0:
                raise ValueError(
                    f"split cutoff {cutoff.isoformat()} falls before the "
                    f"first cleaned bar of {symbol} "
                    f"({ts.iloc[0].isoformat()}); the symbol would have an "
                    f"empty training set. Adjust split_date / train_ratio "
                    f"or drop {symbol} from symbols."
                )
            if split >= n_total:
                raise ValueError(
                    f"split cutoff {cutoff.isoformat()} is at or after the "
                    f"last cleaned bar of {symbol} "
                    f"({ts.iloc[-1].isoformat()}); the symbol would have "
                    f"an empty test set."
                )
            X = df[FEATURE_NAMES].values.astype(np.float32)
            y = df["target"].values.astype(np.float32)
            ref = df["reference"].values.astype(np.float32)
            per_symbol[symbol] = {
                "X_train": X[:split],
                "y_train": y[:split],
                "X_test": X[split:],
                "y_test": y[split:],
                "reference_train": ref[:split],
                "reference_test": ref[split:],
                "feature_names": list(FEATURE_NAMES),
                "df": df,
            }
        self._data = per_symbol
        self._split_cutoff: pd.Timestamp = cutoff
        return self._data

    def _resolve_cutoff(
        self, per_symbol_ts: Dict[str, pd.Series]
    ) -> pd.Timestamp:
        """Decide the inclusive last-train-bar timestamp shared by every symbol.

        Two modes:

        * **Explicit (``split_date`` set)** — use it verbatim. Train rows
          have ``timestamp <= split_date``; everything later is test.
        * **Auto-derived** — pivot on the symbol with the *latest* first
          cleaned bar (i.e. the most-constrained range), apply
          ``train_ratio`` to that symbol's row count, and read off the
          last-train timestamp. Pivot choice means the cutoff is
          invariant to adding more *long-history* symbols (BTC/ETH style)
          and only moves when an even-later-listed symbol enters
          ``symbols`` — which is the correct behavior because that's
          exactly when the held-out window would otherwise drift.
        """
        if self._split_date is not None:
            return self._split_date
        pivot = max(
            self._symbols, key=lambda s: per_symbol_ts[s].iloc[0]
        )
        pivot_ts = per_symbol_ts[pivot]
        pivot_split = int(len(pivot_ts) * self._train_ratio)
        if pivot_split <= 0 or pivot_split >= len(pivot_ts):
            raise ValueError(
                f"train_ratio={self._train_ratio} gives a degenerate split "
                f"on pivot symbol {pivot} (n={len(pivot_ts)}, "
                f"split_index={pivot_split})."
            )
        return pivot_ts.iloc[pivot_split - 1]

    # ForecastingTask hooks

    def get_features(self) -> Dict[str, np.ndarray]:
        self.load_data()
        return {sym: self._data[sym]["X_test"] for sym in self._symbols}

    def get_ground_truth(self) -> Dict[str, np.ndarray]:
        self.load_data()
        return {sym: self._data[sym]["y_test"] for sym in self._symbols}

    # The base ForecastingTask expects ``self._data`` to be a B-shape
    # ``{'train': {...}, 'test': {...}}`` bundle, but our curated layout
    # is ``{sym: {X_train, y_train, X_test, y_test, ...}}``. Override
    # the train accessors so the curated verifier (and any agent calling
    # ``predict_and_evaluate(split='train')``) can read training data.
    def get_train_features(self) -> Dict[str, np.ndarray]:
        self.load_data()
        return {sym: self._data[sym]["X_train"] for sym in self._symbols}

    def get_train_ground_truth(self) -> Dict[str, np.ndarray]:
        self.load_data()
        return {sym: self._data[sym]["y_train"] for sym in self._symbols}

    def get_reference_train(self) -> Dict[str, np.ndarray]:
        """Per-symbol current-bar close prices for the training split.

        Bundled into ``dataset.h5`` so the agent can compute relative
        changes / sanity-check directional signals against absolute
        price predictions.
        """
        self.load_data()
        return {sym: self._data[sym]["reference_train"] for sym in self._symbols}

    def get_reference_test(self) -> Dict[str, np.ndarray]:
        """Per-symbol current-bar close prices for the held-out split.

        Used at scoring time to derive directional accuracy from
        absolute price predictions:
        ``sign(predicted_price[i] - reference_test[i])``.
        """
        self.load_data()
        return {sym: self._data[sym]["reference_test"] for sym in self._symbols}

    @property
    def forecast_horizon_bars(self) -> int:
        """Forecast horizon in bars (read-only, set at construction)."""
        return self._forecast_horizon_bars

    @property
    def headline_metric(self) -> str:
        """Configured headline metric (read-only, set at construction)."""
        return self._headline_metric

    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "per_symbol": {
                "shape": (len(FEATURE_NAMES),),
                "dtype": "float32",
                "features": list(FEATURE_NAMES),
            },
            "symbols": list(self._symbols),
        }

    def get_action_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "per_symbol": {
                "type": "continuous",
                "shape": (1,),
                "description": (
                    "predicted absolute close price "
                    f"{self._forecast_horizon_bars} bar(s) ahead"
                ),
            },
            "symbols": list(self._symbols),
        }

    # Scoring (dict-aware)

    def _default_score(self, predictions: Any, ground_truth: Any) -> Dict[str, float]:
        if isinstance(predictions, dict) and isinstance(ground_truth, dict):
            return self._score_dict(predictions, ground_truth)
        return super()._default_score(predictions, ground_truth)

    def _score_dict(
        self,
        predictions: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> Dict[str, float]:
        """Score price predictions through the gt-at-init Loss panel.

        Per-symbol metrics: ``mse``, ``rmse``, ``mae``, ``mape``,
        ``r2``, ``pearson`` (computed via the existing torch
        :class:`Loss` classes — single source of truth shared with the
        auto-pipeline assembled evaluator) plus ``directional_accuracy``
        (derived from sign of price minus reference). Macro keys
        (``mape``, ``r2``, ``pearson``, ``directional_accuracy``) are
        the unweighted means across :attr:`_target_symbols`. The
        ``reward`` key is whichever macro the task was configured for
        via ``headline_metric``.
        """
        self.load_data()
        out: Dict[str, float] = {}
        target_set = set(self._target_symbols)
        for sym, gt_raw in ground_truth.items():
            pred_raw = predictions.get(sym)
            gt_arr = np.asarray(gt_raw).astype(float).flatten()
            n_gt = len(gt_arr)
            if pred_raw is None or n_gt == 0:
                # Empty predictions or empty ground truth: emit zeros so
                # downstream callers get a stable key set, rather than
                # crashing on a missing key. A zero MAPE is a misleading
                # "perfect score" but the dry-run probe in on_startup
                # explicitly invokes this path with zeros, so the
                # behaviour is internally consistent.
                for name, _ in _PANEL_LOSS_CLASSES:
                    out[f"{name}_{sym}"] = 0.0
                out[f"directional_accuracy_{sym}"] = 0.0
                continue
            pred_arr = np.asarray(pred_raw).astype(float).flatten()
            n = min(len(pred_arr), n_gt)
            pred_arr = pred_arr[:n]
            gt_arr = gt_arr[:n]
            pred_t = torch.from_numpy(pred_arr.astype(np.float32))
            gt_t = torch.from_numpy(gt_arr.astype(np.float32))
            for name, cls in _PANEL_LOSS_CLASSES:
                # gt-at-init pattern (RULE A): construct per-call with
                # the symbol's gt window, then forward(pred). Lightweight
                # — just stores a tensor reference.
                out[f"{name}_{sym}"] = float(cls(gt=gt_t).forward(pred_t))
            # Directional accuracy via signed (price - reference) diff.
            # Reference is the current-bar close at row t; without it,
            # `sign(predicted_price)` would always be positive for
            # absolute prices and the metric collapses.
            ref_full = np.asarray(
                self._data[sym]["reference_test"]
            ).astype(float).flatten()
            ref = ref_full[:n]
            ref_t = torch.from_numpy(ref.astype(np.float32))
            out[f"directional_accuracy_{sym}"] = float(
                DirectionalAccuracy(gt=(gt_t - ref_t)).forward(pred_t - ref_t)
            )
        # Macro-aggregate scale-free metrics over target_symbols only.
        # Per-symbol absolute MSE/MAE/RMSE are emitted but NOT macro-
        # averaged because cross-symbol price scales differ (BTCUSDT ~
        # 60k vs SPY ~ 500); the macro mean would just track the widest-
        # range symbol.
        for name in _AGG_KEYS:
            vals = [
                out[f"{name}_{s}"] for s in self._target_symbols
                if f"{name}_{s}" in out
            ]
            out[name] = sum(vals) / len(vals) if vals else 0.0
        # Surface the configured headline as the canonical ``reward``
        # key. coerce_scores in the curated handler picks this up
        # directly without needing the headline name in the spec.
        out["reward"] = out[self._headline_metric]
        return out

    # Gym-loop hooks (overridden for dict features)

    def _get_num_samples(self) -> int:
        self.load_data()
        first = self._symbols[0]
        return int(len(self._data[first]["X_test"]))

    def _get_observation_at(self, idx: int) -> Dict[str, Any]:
        self.load_data()
        out: Dict[str, Any] = {}
        for sym in self._symbols:
            arr = self._data[sym]["X_test"]
            if 0 <= idx < len(arr):
                out[sym] = arr[idx]
        return out
