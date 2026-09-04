"""Offline and realtime smoke tests for the curated task wrappers."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Make curated task packages importable.
_TASKS_DIR = (
    Path(__file__).resolve().parent.parent / "tasks"
)
if str(_TASKS_DIR) not in sys.path:
    sys.path.insert(0, str(_TASKS_DIR))

from openfinai_pipeline.benchmark.contracts import BaseAgent, TaskMetadata
from openfinai_pipeline.realtime.data_providers.base import MarketSnapshot
from openfinai_pipeline.realtime.ledger import PredictionLedger


# Mock providers


class _OfflineMockProvider:
    """120-bar deterministic synthetic series for offline load_data tests."""

    name = "mock"

    def get_current_price(self, symbol: str) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            price=100.0,
        )

    def get_price_at(self, symbol: str, at: datetime) -> MarketSnapshot:
        return MarketSnapshot(symbol=symbol, timestamp=at, price=100.0)

    def get_bars(self, symbol, interval, start, end):
        bars = []
        for i in range(120):
            p = 100.0 + (i % 10)
            bars.append(
                MarketSnapshot(
                    symbol=symbol,
                    timestamp=start + timedelta(hours=i),
                    price=p,
                    open=p,
                    high=p + 0.5,
                    low=p - 0.5,
                    close=p,
                    volume=10.0 + i,
                )
            )
        return bars


class _LiveMockProvider:
    """Five-bar provider for live-task smoke tests."""

    name = "mock"

    def __init__(self, price: float = 100.0) -> None:
        self._price = price

    def get_current_price(self, symbol: str) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            price=self._price,
        )

    def get_price_at(self, symbol: str, at: datetime) -> MarketSnapshot:
        return MarketSnapshot(symbol=symbol, timestamp=at, price=self._price)

    def get_bars(self, symbol, interval, start, end):
        bars = []
        for i in range(5):
            bars.append(
                MarketSnapshot(
                    symbol=symbol,
                    timestamp=start + timedelta(minutes=i),
                    price=self._price + i,
                )
            )
        return bars


_RealtimeMockProvider = _LiveMockProvider


# Level A: import smoke


CURATED_TASK_MODULES = [
    "offline_crypto_forecasting",
    "offline_crypto_trading",
    "offline_stock_forecasting",
    "offline_stock_trading",
    "realtime_crypto_forecasting",
    "realtime_crypto_trading",
    "realtime_stock_forecasting",
    "realtime_stock_trading",
    "realtime_stock_trading_alpaca_paper",
]


@pytest.mark.parametrize("module_name", CURATED_TASK_MODULES)
def test_curated_task_module_imports(module_name: str) -> None:
    """Each curated task package imports without ModuleNotFoundError."""
    import importlib

    importlib.import_module(module_name)


# Level B: offline tasks load_data + scoring


@pytest.fixture
def synthetic_offline_kwargs() -> dict[str, Any]:
    """Far-future date range so the cache CSV path is unique per test run."""
    return {
        "symbols": ["TEST"],
        "start": "2099-01-01",
        "end": "2099-01-10",
    }


def test_offline_crypto_forecasting_load(synthetic_offline_kwargs):
    from offline_crypto_forecasting import OfflineCryptoForecasting, FEATURE_NAMES

    task = OfflineCryptoForecasting(
        config=synthetic_offline_kwargs,
        provider=_OfflineMockProvider(),
    )
    meta = task.metadata()
    assert meta.task_id.startswith("offline_crypto_forecasting_")
    assert meta.interaction_model == "forecasting"

    feats = task.get_features()
    assert isinstance(feats, dict)
    assert "TEST" in feats
    assert feats["TEST"].shape[1] == len(FEATURE_NAMES)
    assert feats["TEST"].dtype == np.float32
    assert feats["TEST"].shape[0] > 0

    gt = task.get_ground_truth()
    assert "TEST" in gt and gt["TEST"].shape == feats["TEST"].shape[:1]

    preds = {"TEST": np.zeros_like(gt["TEST"])}
    score = task.predict_and_evaluate(preds)
    assert "directional_accuracy" in score
    assert "directional_accuracy_TEST" in score


def test_offline_stock_forecasting_load(synthetic_offline_kwargs):
    from offline_stock_forecasting import OfflineStockForecasting, FEATURE_NAMES

    task = OfflineStockForecasting(
        config=synthetic_offline_kwargs,
        provider=_OfflineMockProvider(),
    )
    meta = task.metadata()
    assert meta.task_id.startswith("offline_stock_forecasting_")
    feats = task.get_features()
    assert "TEST" in feats
    assert feats["TEST"].shape[1] == len(FEATURE_NAMES)
    score = task.predict_and_evaluate({"TEST": np.zeros_like(task.get_ground_truth()["TEST"])})
    assert "directional_accuracy" in score


def test_offline_crypto_trading_episode(synthetic_offline_kwargs):
    from offline_crypto_trading import OfflineCryptoTrading

    cfg = dict(synthetic_offline_kwargs)
    cfg["episode_length"] = 5
    task = OfflineCryptoTrading(config=cfg, provider=_OfflineMockProvider())
    assert task.metadata().task_id.startswith("offline_crypto_trading_")
    obs = task.reset()
    assert "symbols" in obs and "TEST" in obs["symbols"]
    assert obs["portfolio"]["positions"].get("TEST", 0.0) == 0.0

    obs2, reward, done, info = task.step(
        {"action": "buy", "symbol": "TEST", "quantity": 1.0}
    )
    assert obs2["portfolio"]["positions"]["TEST"] == 1.0
    assert "fills" in info

    rewards = task.evaluate([])
    for k in ("total_return", "num_trades", "sharpe_ratio", "max_drawdown"):
        assert k in rewards


def test_offline_stock_trading_episode(synthetic_offline_kwargs):
    from offline_stock_trading import OfflineStockTrading

    cfg = dict(synthetic_offline_kwargs)
    cfg["episode_length"] = 5
    task = OfflineStockTrading(config=cfg, provider=_OfflineMockProvider())
    assert task.metadata().task_id.startswith("offline_stock_trading_")
    obs = task.reset()
    assert obs["portfolio"]["positions"].get("TEST", 0.0) == 0.0
    obs2, reward, done, info = task.step(
        {"action": "sell", "symbol": "TEST", "quantity": 1.0}
    )
    assert obs2["portfolio"]["positions"]["TEST"] == -1.0
    rewards = task.evaluate([])
    assert "total_return" in rewards


def test_offline_crypto_trading_multi_symbol(synthetic_offline_kwargs):
    from offline_crypto_trading import OfflineCryptoTrading

    cfg = dict(synthetic_offline_kwargs)
    cfg["symbols"] = ["A", "B"]
    cfg["episode_length"] = 5
    task = OfflineCryptoTrading(config=cfg, provider=_OfflineMockProvider())
    obs = task.reset()
    assert set(obs["symbols"].keys()) == {"A", "B"}
    obs2, reward, done, info = task.step({
        "orders": [
            {"action": "buy", "symbol": "A", "quantity": 1.0},
            {"action": "sell", "symbol": "B", "quantity": 1.0},
        ]
    })
    assert obs2["portfolio"]["positions"] == {"A": 1.0, "B": -1.0}


def test_offline_stock_forecasting_multi_symbol(synthetic_offline_kwargs):
    from offline_stock_forecasting import OfflineStockForecasting

    cfg = dict(synthetic_offline_kwargs)
    cfg["symbols"] = ["A", "B"]
    task = OfflineStockForecasting(config=cfg, provider=_OfflineMockProvider())
    feats = task.get_features()
    assert set(feats.keys()) == {"A", "B"}
    assert feats["A"].shape == feats["B"].shape


# Level C: live tasks with mock provider


def test_live_crypto_forecasting_step():
    from realtime_crypto_forecasting import RealtimeCryptoForecasting

    task = RealtimeCryptoForecasting(
        config={"symbols": ["BTCUSDT"], "horizon_bars": 5, "db_path": ":memory:"},
        provider=_LiveMockProvider(),
        ledger=PredictionLedger(":memory:"),
    )
    meta = task.metadata()
    assert meta.task_id == "realtime_crypto_forecasting_btcusdt"
    obs = task.reset()
    assert "BTCUSDT" in obs
    obs2, reward, done, info = task.step(
        {"symbol": "BTCUSDT", "direction": "long"}
    )
    assert reward == 0.0  # ground truth deferred
    assert "prediction_id" in info or info.get("done", False) is False


def test_live_stock_forecasting_step():
    from realtime_stock_forecasting import RealtimeStockForecasting

    task = RealtimeStockForecasting(
        config={"symbols": ["SPY"], "horizon_bars": 5, "db_path": ":memory:"},
        provider=_LiveMockProvider(),
        ledger=PredictionLedger(":memory:"),
    )
    meta = task.metadata()
    assert meta.task_id == "realtime_stock_forecasting_spy"
    obs = task.reset()
    assert "SPY" in obs
    obs2, reward, done, info = task.step(
        {"symbol": "SPY", "direction": "short"}
    )
    assert reward == 0.0


def test_live_crypto_trading_step():
    from realtime_crypto_trading import RealtimeCryptoTrading

    task = RealtimeCryptoTrading(
        config={"symbols": ["BTCUSDT"], "max_steps": 1},
        provider=_LiveMockProvider(),
    )
    meta = task.metadata()
    assert meta.task_id == "realtime_crypto_trading_btcusdt"
    obs = task.reset()
    assert "BTCUSDT" in obs["symbols"]
    obs2, reward, done, info = task.step(
        {"action": "buy", "symbol": "BTCUSDT", "quantity": 1.0}
    )
    assert done is True  # max_steps=1
    assert "total_pnl" in info


def test_live_stock_trading_step():
    from realtime_stock_trading import RealtimeStockTrading

    task = RealtimeStockTrading(
        config={"symbols": ["SPY"], "max_steps": 1},
        provider=_LiveMockProvider(),
    )
    meta = task.metadata()
    assert meta.task_id == "realtime_stock_trading_spy"
    obs = task.reset()
    assert "SPY" in obs["symbols"]
    obs2, reward, done, info = task.step(
        {"action": "buy", "symbol": "SPY", "quantity": 1.0}
    )
    assert done is True
    assert "total_pnl" in info


class _AlpacaNameMockProvider(_LiveMockProvider):
    """Mock provider that claims to be Alpaca so alpaca_paper executor
    routing accepts it. Does not implement any real Alpaca endpoints."""

    name = "alpaca"


def test_live_stock_trading_alpaca_paper_metadata(monkeypatch):
    """Verify the alpaca_paper task instantiates and carries the right
    metadata. We do NOT call ``reset()`` or ``step()`` because both
    would hit paper-api.alpaca.markets via AlpacaPaperExecutor.
    """
    from openfinai_pipeline.realtime.execution import alpaca_paper as alpaca_paper_mod

    # Stub AlpacaPaperExecutor so no network / real auth is needed.
    # Mirrors the real executor's constructor signature (positional
    # config args + the flatten_on_start kwarg forwarded by
    # create_executor) so this stub doesn't break when those evolve.
    class _StubExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self._positions: dict[str, float] = {}

        def execute(self, *a, **kw):  # pragma: no cover - unused here
            raise NotImplementedError

        def get_positions(self):
            return dict(self._positions)

        def compute_pnl(self, current_prices):
            return {"__total": 0.0}

        def get_trade_log(self):
            return []

        def reset(self) -> None:
            self._positions.clear()

    monkeypatch.setattr(alpaca_paper_mod, "AlpacaPaperExecutor", _StubExecutor)

    from realtime_stock_trading_alpaca_paper import RealtimeStockTradingAlpacaPaper

    task = RealtimeStockTradingAlpacaPaper(
        config={"symbols": ["SPY"], "max_steps": 1},
        provider=_AlpacaNameMockProvider(),
    )
    meta = task.metadata()
    assert meta.task_id == "realtime_stock_trading_alpaca_paper_spy"
    assert "alpaca_paper" in (meta.tags or [])
    assert task._trading_config.execution_mode == "alpaca_paper"


def test_live_stock_trading_alpaca_paper_rejects_non_alpaca_provider():
    """A non-Alpaca provider should fail at executor construction."""
    from realtime_stock_trading_alpaca_paper import RealtimeStockTradingAlpacaPaper

    with pytest.raises(ValueError, match="alpaca_paper"):
        RealtimeStockTradingAlpacaPaper(
            config={"symbols": ["SPY"]},
            provider=_LiveMockProvider(),  # name="mock"
        )


# Level D: target_symbols + extra_resolutions


def test_target_symbols_must_be_subset(synthetic_offline_kwargs):
    """target_symbols must be a subset of symbols; rogue entry rejected."""
    from offline_crypto_trading import OfflineCryptoTrading

    cfg = dict(synthetic_offline_kwargs)
    cfg["symbols"] = ["A", "B"]
    cfg["target_symbols"] = ["A", "Z"]  # Z is not in symbols
    cfg["episode_length"] = 5
    with pytest.raises(ValueError, match="subset of symbols"):
        OfflineCryptoTrading(config=cfg, provider=_OfflineMockProvider())


def test_context_resolutions_rejects_duplicate_interval(synthetic_offline_kwargs):
    """context_resolutions intervals must be unique."""
    from offline_crypto_trading import OfflineCryptoTrading

    cfg = dict(synthetic_offline_kwargs)
    cfg["context_resolutions"] = [
        {"interval": "1h", "bars": 5},
        {"interval": "1h", "bars": 10},
    ]
    cfg["data_resolution"] = "1h"
    with pytest.raises(ValueError, match="more than once"):
        OfflineCryptoTrading(config=cfg, provider=_OfflineMockProvider())


def test_context_resolutions_rejects_bad_shape(synthetic_offline_kwargs):
    """context_resolutions entries must be {interval, bars}."""
    from offline_crypto_trading import OfflineCryptoTrading

    cfg = dict(synthetic_offline_kwargs)
    cfg["context_resolutions"] = [{"interval": "4h"}]  # missing bars
    with pytest.raises(ValueError, match="bars"):
        OfflineCryptoTrading(config=cfg, provider=_OfflineMockProvider())


def test_context_resolutions_step_required_when_multi(synthetic_offline_kwargs):
    """data_resolution is required when context_resolutions has 2+ entries."""
    from offline_crypto_trading import OfflineCryptoTrading

    cfg = dict(synthetic_offline_kwargs)
    cfg["context_resolutions"] = [
        {"interval": "1h", "bars": 5},
        {"interval": "4h", "bars": 6},
    ]
    # data_resolution intentionally omitted
    with pytest.raises(ValueError, match="data_resolution must be specified"):
        OfflineCryptoTrading(config=cfg, provider=_OfflineMockProvider())


def test_context_resolutions_step_must_match_entry(synthetic_offline_kwargs):
    """data_resolution must match an interval present in the list."""
    from offline_crypto_trading import OfflineCryptoTrading

    cfg = dict(synthetic_offline_kwargs)
    cfg["context_resolutions"] = [
        {"interval": "1h", "bars": 5},
        {"interval": "4h", "bars": 6},
    ]
    cfg["data_resolution"] = "1d"  # not in the list
    with pytest.raises(ValueError, match="does not match any entry"):
        OfflineCryptoTrading(config=cfg, provider=_OfflineMockProvider())


def test_data_resolution_must_be_finest(synthetic_offline_kwargs):
    """data_resolution must be strictly finer than every sidecar."""
    from offline_crypto_trading import OfflineCryptoTrading

    cfg = dict(synthetic_offline_kwargs)
    cfg["context_resolutions"] = [
        {"interval": "1h", "bars": 24},  # coarser
        {"interval": "4h", "bars": 6},   # would-be primary
    ]
    cfg["data_resolution"] = "4h"  # 4h > 1h, so 4h is NOT the finest
    with pytest.raises(ValueError, match="not finer than"):
        OfflineCryptoTrading(config=cfg, provider=_OfflineMockProvider())


def test_context_resolutions_single_entry_no_step_required(
    synthetic_offline_kwargs,
):
    """When the list has exactly one entry, data_resolution is optional."""
    from offline_crypto_trading import OfflineCryptoTrading

    cfg = dict(synthetic_offline_kwargs)
    cfg["context_resolutions"] = [{"interval": "1h", "bars": 5}]
    cfg["episode_length"] = 3
    # No data_resolution; should not raise
    task = OfflineCryptoTrading(config=cfg, provider=_OfflineMockProvider())
    assert task._interval == "1h"
    assert task._context_bars == 5
    assert task._extra_resolutions == ()


def test_offline_crypto_trading_target_symbols_filters_eval(
    synthetic_offline_kwargs,
):
    """When target_symbols is a strict subset, evaluate() reflects only
    target P&L; non-target trades stay in trade_history but don't feed
    the reward bank."""
    from offline_crypto_trading import OfflineCryptoTrading

    cfg = dict(synthetic_offline_kwargs)
    cfg["symbols"] = ["A", "B"]
    cfg["target_symbols"] = ["A"]
    cfg["episode_length"] = 5
    task = OfflineCryptoTrading(config=cfg, provider=_OfflineMockProvider())
    task.reset()
    # Drive a few steps with non-zero positions on both symbols
    for _ in range(3):
        task.step({
            "orders": [
                {"action": "buy", "symbol": "A", "quantity": 1.0},
                {"action": "sell", "symbol": "B", "quantity": 1.0},
            ]
        })

    # Target-restricted total_return equals the sum of A's per-step
    # contributions only (per_symbol_rewards is on the step rows; audit
    # ``kind`` rows are skipped).
    metrics = task.evaluate([])
    step_rows = [e for e in task._trade_history if "kind" not in e]
    target_total = sum(
        e.get("per_symbol_rewards", {}).get("A", 0.0) for e in step_rows
    )
    assert metrics["total_return"] == pytest.approx(target_total)
    # Non-target return is excluded
    full_total = sum(e["reward"] for e in step_rows)
    if abs(full_total - target_total) > 1e-9:
        assert metrics["total_return"] != pytest.approx(full_total)


def test_offline_crypto_trading_extra_resolutions_observation(
    synthetic_offline_kwargs,
):
    """Observation grows recent_bars_by_interval when 2+ resolutions configured."""
    from offline_crypto_trading import OfflineCryptoTrading

    cfg = dict(synthetic_offline_kwargs)
    cfg["context_resolutions"] = [
        {"interval": "1h", "bars": 20},
        {"interval": "4h", "bars": 6},
        {"interval": "1d", "bars": 3},
    ]
    cfg["data_resolution"] = "1h"
    cfg["episode_length"] = 3
    task = OfflineCryptoTrading(config=cfg, provider=_OfflineMockProvider())
    obs = task.reset()
    sym_obs = obs["symbols"]["TEST"]
    assert "recent_bars_by_interval" in sym_obs
    by_interval = sym_obs["recent_bars_by_interval"]
    # Step resolution + both sidecars present
    assert set(by_interval.keys()) == {"1h", "4h", "1d"}
    # Each sidecar slice has at most `bars` entries
    assert len(by_interval["4h"]) <= 6
    assert len(by_interval["1d"]) <= 3
    # And the step-resolution slice respects its bars
    assert len(by_interval["1h"]) <= 20


def test_offline_stock_forecasting_target_symbols_aggregate(
    synthetic_offline_kwargs,
):
    """directional_accuracy aggregates only over target symbols."""
    from offline_stock_forecasting import OfflineStockForecasting

    cfg = dict(synthetic_offline_kwargs)
    cfg["symbols"] = ["A", "B"]
    cfg["target_symbols"] = ["A"]
    task = OfflineStockForecasting(config=cfg, provider=_OfflineMockProvider())
    feats = task.get_features()
    gt = task.get_ground_truth()
    # Predictions: B always wrong (negate gt), A always right (mirror gt)
    preds = {sym: gt[sym] for sym in feats}
    preds["B"] = -gt["B"]
    score = task.predict_and_evaluate(preds)
    # Per-symbol scores still emitted for both
    assert "directional_accuracy_A" in score
    assert "directional_accuracy_B" in score
    # B's bad accuracy must NOT drag the aggregate (target is A only).
    assert score["directional_accuracy"] == pytest.approx(
        score["directional_accuracy_A"]
    )


def test_realtime_forecasting_drops_off_target_predictions(monkeypatch):
    """step() drops predictions for symbols outside target_symbols and
    surfaces dropped_off_target in the info dict — no ledger write."""
    from realtime_crypto_forecasting import RealtimeCryptoForecasting

    ledger = PredictionLedger(":memory:")
    task = RealtimeCryptoForecasting(
        config={
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "target_symbols": ["BTCUSDT"],
            "horizon_bars": 5,
            "db_path": ":memory:",
        },
        provider=_RealtimeMockProvider(),
        ledger=ledger,
    )
    task.reset()
    # Off-target prediction
    obs2, reward, done, info = task.step(
        {"symbol": "ETHUSDT", "direction": "long"}
    )
    assert info.get("dropped_off_target") is True
    assert "prediction_id" not in info
    assert reward == 0.0
    # On-target prediction goes through normally
    obs3, reward2, done2, info2 = task.step(
        {"symbol": "BTCUSDT", "direction": "long"}
    )
    assert "prediction_id" in info2
    assert info2.get("dropped_off_target") is None


def test_realtime_forecasting_extra_resolutions_observation():
    """Observation grows recent_bars_by_interval when 2+ resolutions configured."""
    from realtime_crypto_forecasting import RealtimeCryptoForecasting

    task = RealtimeCryptoForecasting(
        config={
            "symbols": ["BTCUSDT"],
            "horizon_bars": 5,
            "db_path": ":memory:",
            "context_resolutions": [
                {"interval": "1m", "bars": 60},
                {"interval": "5m", "bars": 4},
                {"interval": "1h", "bars": 2},
            ],
            "data_resolution": "1m",
        },
        provider=_RealtimeMockProvider(),
        ledger=PredictionLedger(":memory:"),
    )
    obs = task.reset()
    sym_obs = obs["BTCUSDT"]
    assert "recent_bars_by_interval" in sym_obs
    by_interval = sym_obs["recent_bars_by_interval"]
    assert set(by_interval.keys()) == {"1m", "5m", "1h"}


def test_realtime_trading_target_symbols_filters_eval():
    """RealtimeTradingTask.evaluate() filters to target_symbols."""
    from realtime_crypto_trading import RealtimeCryptoTrading

    task = RealtimeCryptoTrading(
        config={
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "target_symbols": ["BTCUSDT"],
            "max_steps": 3,
        },
        provider=_RealtimeMockProvider(),
    )
    task.reset()
    task.step({"action": "buy", "symbol": "BTCUSDT", "quantity": 1.0})
    task.step({"action": "buy", "symbol": "ETHUSDT", "quantity": 1.0})
    rewards = task.evaluate([])
    # Headline total_return is scoped to the target (BTCUSDT) via the
    # per-symbol reward attribution, even though both symbols traded.
    step_rows = [t for t in task._trade_history if "kind" not in t]
    expected = sum(
        r.get("per_symbol_rewards", {}).get("BTCUSDT", 0.0) for r in step_rows
    )
    assert rewards["total_return"] == pytest.approx(expected)


def test_realtime_trading_extra_resolutions_observation():
    """Realtime trading observation surfaces recent_bars_by_interval."""
    from realtime_crypto_trading import RealtimeCryptoTrading

    task = RealtimeCryptoTrading(
        config={
            "symbols": ["BTCUSDT"],
            "max_steps": 1,
            "context_resolutions": [
                {"interval": "1m", "bars": 60},
                {"interval": "5m", "bars": 4},
                {"interval": "1h", "bars": 2},
            ],
            "data_resolution": "1m",
        },
        provider=_RealtimeMockProvider(),
    )
    obs = task.reset()
    sym_obs = obs["symbols"]["BTCUSDT"]
    assert "recent_bars_by_interval" in sym_obs
    by_interval = sym_obs["recent_bars_by_interval"]
    assert set(by_interval.keys()) == {"1m", "5m", "1h"}


# Level E: downsample helper + auto-scaled lookback


def test_downsample_bars_aggregates_ohlcv():
    """5 consecutive 1m bars aggregate into 1 5m bar with correct OHLCV."""
    from openfinai_pipeline.realtime.data_providers.base import (
        MarketSnapshot,
        downsample_bars,
    )

    base = datetime(2026, 5, 9, 14, 0, 0, tzinfo=timezone.utc)
    primary = []
    # 5 1m bars covering a single 5m bucket [14:00, 14:05)
    # Prices: 100, 102, 99, 105, 103
    closes = [100.0, 102.0, 99.0, 105.0, 103.0]
    for i, c in enumerate(closes):
        primary.append(
            MarketSnapshot(
                symbol="X",
                timestamp=base + timedelta(minutes=i),
                price=c,
                open=closes[max(0, i - 1)] if i else c,
                high=c + 1.0,
                low=c - 1.0,
                close=c,
                volume=10.0,
            )
        )
    out = downsample_bars(primary, "5m", 1)
    assert len(out) == 1
    bar = out[0]
    # Bucket-aligned to 14:00:00
    assert bar.timestamp == base
    # open = first bar's open
    assert bar.open == primary[0].open
    # close = last bar's close
    assert bar.close == primary[-1].close
    # high = max over the bucket
    assert bar.high == max(b.high for b in primary)
    # low = min over the bucket
    assert bar.low == min(b.low for b in primary)
    # volume = sum
    assert bar.volume == 50.0


def test_downsample_bars_clock_aligned_buckets():
    """Bars across a 5m boundary land in two distinct buckets."""
    from openfinai_pipeline.realtime.data_providers.base import (
        MarketSnapshot,
        downsample_bars,
    )

    # Bars at 14:03, 14:04, 14:05, 14:06 — first two go into 14:00
    # bucket, last two into 14:05 bucket.
    base = datetime(2026, 5, 9, 14, 3, 0, tzinfo=timezone.utc)
    primary = [
        MarketSnapshot(
            symbol="X",
            timestamp=base + timedelta(minutes=i),
            price=100.0 + i,
        )
        for i in range(4)
    ]
    out = downsample_bars(primary, "5m", 5)
    assert len(out) == 2
    # First bucket: 14:00:00, contains bars at 14:03 and 14:04
    assert out[0].timestamp == datetime(
        2026, 5, 9, 14, 0, 0, tzinfo=timezone.utc
    )
    # Second bucket: 14:05:00, contains bars at 14:05 and 14:06
    assert out[1].timestamp == datetime(
        2026, 5, 9, 14, 5, 0, tzinfo=timezone.utc
    )


def test_downsample_bars_returns_last_n():
    """Returns at most `target_bars` aggregated buckets, latest-first-trimmed."""
    from openfinai_pipeline.realtime.data_providers.base import (
        MarketSnapshot,
        downsample_bars,
    )

    # 30 1m bars across 6 5m buckets
    base = datetime(2026, 5, 9, 14, 0, 0, tzinfo=timezone.utc)
    primary = [
        MarketSnapshot(
            symbol="X", timestamp=base + timedelta(minutes=i), price=100.0
        )
        for i in range(30)
    ]
    out = downsample_bars(primary, "5m", 3)
    assert len(out) == 3
    # Should be the last 3 buckets: 14:15, 14:20, 14:25
    assert [b.timestamp.minute for b in out] == [15, 20, 25]


def test_realtime_trading_sidecar_stays_fresh_after_steps():
    """Sidecar windows reflect new primary-buffer ticks."""
    from realtime_crypto_trading import RealtimeCryptoTrading

    # Keep live ticks distinct from the 100–104 backfill range.
    class _StreamingMockProvider:
        name = "mock"

        def __init__(self) -> None:
            self._tick = 0

        def get_current_price(self, symbol):
            self._tick += 1
            return MarketSnapshot(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                price=1000.0 + self._tick,
            )

        def get_price_at(self, symbol, at):
            return MarketSnapshot(symbol=symbol, timestamp=at, price=100.0)

        def get_bars(self, symbol, interval, start, end):
            return [
                MarketSnapshot(
                    symbol=symbol,
                    timestamp=start + timedelta(minutes=i),
                    price=100.0 + i,
                    open=100.0 + i,
                    high=100.0 + i,
                    low=100.0 + i,
                    close=100.0 + i,
                    volume=1.0,
                )
                for i in range(5)
            ]

    task = RealtimeCryptoTrading(
        config={
            "symbols": ["BTCUSDT"],
            "max_steps": 0,  # unlimited
            "context_resolutions": [
                {"interval": "1m", "bars": 5},
                {"interval": "5m", "bars": 2},
            ],
            "data_resolution": "1m",
        },
        provider=_StreamingMockProvider(),
    )
    # Mock provider has no WS; force REST every step so the per-step
    # tick-price increments propagate to the buffer (the freshness
    # this test is verifying).
    task.PRICE_STALENESS_TTL_MS = 0.0
    task.reset()
    # Drive 3 steps; each inserts a new snapshot
    for _ in range(3):
        obs, _, _, _ = task.step(
            {"action": "hold", "symbol": "BTCUSDT", "quantity": 0.0}
        )
    sym_obs = obs["symbols"]["BTCUSDT"]
    by_interval = sym_obs["recent_bars_by_interval"]
    # Sidecar 5m view must reflect the latest tick. Compare against the
    # latest 1m bar's `price` (not `close`) because get_current_price
    # snapshots only populate `price`; the downsampler falls back to
    # `price` when `close` is None, so the 5m bucket's close should
    # match the latest 1m snapshot's price.
    last_5m_close = by_interval["5m"][-1].close
    last_1m_price = by_interval["1m"][-1].price
    assert last_5m_close == last_1m_price
    # Live-tick prices start at 1000+ in the mock, so any value above
    # the backfill ceiling (104.0) proves the sidecar isn't stuck on the
    # backfill cache. Use a generous threshold (>= 1001) so the
    # assertion is robust to either fetch cadence.
    assert last_5m_close >= 1001.0


def test_auto_scaled_lookback_matches_deepest_sidecar():
    """primary_lookback = max(step_bars, max(ex_bars * ratio))."""
    from realtime_crypto_trading import RealtimeCryptoTrading

    # Sidecar 1h × 24 bars => 24 * 60 = 1440 1m primary bars needed
    task = RealtimeCryptoTrading(
        config={
            "symbols": ["BTCUSDT"],
            "max_steps": 1,
            "context_resolutions": [
                {"interval": "1m", "bars": 60},
                {"interval": "1h", "bars": 24},
            ],
            "data_resolution": "1m",
        },
        provider=_RealtimeMockProvider(),
    )
    assert task._lookback_bars == 1440
