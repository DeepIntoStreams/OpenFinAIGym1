"""Offline/realtime trading parity on an identical zero-friction price tape.

Mocks force deterministic REST snapshots and bypass external data access.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from openfinai_pipeline.settings import TradingConfig
from openfinai_pipeline.realtime.data_providers.base import MarketSnapshot
from openfinai_pipeline.realtime.tasks.realtime_trading_task import (
    RealtimeTradingTask,
)
from tasks.offline_crypto_trading.task import OfflineCryptoTrading

_CACHE_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "pipeline_output"
    / "datasets"
    / "binance_crypto_hourly_ohlcv"
)

# Shared reward-stream metrics; task-specific counts and total_pnl are excluded.
_SHARED_METRIC_KEYS = (
    "total_return",
    "mean_return",
    "std_return",
    "pnl",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
)


class _TapeProvider:
    """Mock provider whose latest price is driven step-by-step."""

    name = "mock"

    def __init__(self, price: float) -> None:
        self._price = float(price)

    def get_current_price(self, symbol: str) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            price=self._price,
        )

    def get_price_at(self, symbol: str, at: datetime) -> MarketSnapshot:
        return MarketSnapshot(symbol=symbol, timestamp=at, price=self._price)

    def get_bars(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[MarketSnapshot]:
        # A flat backfill window; only the live latest price matters for PnL
        # (compute_pnl iterates the fill log, not the backfill bars).
        return [
            MarketSnapshot(
                symbol=symbol,
                timestamp=start + timedelta(minutes=i),
                price=self._price,
            )
            for i in range(5)
        ]

    def set_price(self, price: float) -> None:
        self._price = float(price)


def _write_offline_tape(symbol: str, closes: list[float], start: str, end: str) -> Path:
    """Write a deterministic OHLCV CSV (flat O/H/L == close) to the cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{symbol}_1h_{start}_{end}.csv"
    rows = [
        {
            "timestamp": f"2020-01-01T{i:02d}:00:00+00:00",
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": 1000.0,
        }
        for i, c in enumerate(closes)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run_offline(closes: list[float], actions: list[Any]) -> tuple[list[float], dict]:
    assert len(closes) == len(actions)
    symbol = "BTCUSDT"
    start, end = "2020-01-01", "2020-01-02"
    # Offline marks step t at close[t-1] and `_episode_end` caps at
    # len(bars)-1, so write one extra (never-stepped) bar to allow exactly
    # len(actions) steps. The pad value is irrelevant (never marked).
    path = _write_offline_tape(symbol, closes + [closes[-1]], start, end)
    try:
        task = OfflineCryptoTrading(
            config={
                "symbols": [symbol],
                "start": start,
                "end": end,
                "initial_cash": 100000.0,
                "slippage_pct": 0.0,
                "transaction_cost_pct": 0.0,
                "episode_length": len(actions),
            }
        )
        task.reset()
        rewards: list[float] = []
        for act in actions:
            _obs, reward, _done, _info = task.step(act)
            rewards.append(float(reward))
        metrics = task.evaluate([])
        return rewards, metrics
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _run_realtime(tape: list[float], actions: list[Any]) -> tuple[list[float], dict]:
    assert len(tape) == len(actions)
    provider = _TapeProvider(price=tape[0])
    task = RealtimeTradingTask(
        provider=provider,
        symbols=["BTCUSDT"],
        trading_config=TradingConfig(slippage_pct=0.0, transaction_cost_pct=0.0),
        initial_capital=100000.0,
        max_steps=len(actions),
    )
    # Force the per-step REST refresh so set_price() propagates into the
    # buffer (the mock has no WebSocket).
    task.PRICE_STALENESS_TTL_MS = 0.0
    task.reset()
    rewards: list[float] = []
    for price, act in zip(tape, actions):
        provider.set_price(price)
        _obs, reward, _done, _info = task.step(act)
        rewards.append(float(reward))
    metrics = task.evaluate([])
    return rewards, metrics


# A buy-and-hold tape: enter at the first bar, then ride the price.
_TAPE = [100.0, 110.0, 105.0, 120.0, 115.0]
_BUY_THEN_HOLD = [
    {"action": "buy", "symbol": "BTCUSDT", "quantity": 1.0},
    {"action": "hold"},
    {"action": "hold"},
    {"action": "hold"},
    {"action": "hold"},
]


def test_offline_realtime_per_step_reward_parity():
    """Per-step rewards match step-for-step at zero friction."""
    off_rewards, _ = _run_offline(_TAPE, _BUY_THEN_HOLD)
    rt_rewards, _ = _run_realtime(_TAPE, _BUY_THEN_HOLD)
    assert off_rewards == pytest.approx(rt_rewards, abs=1e-9)
    # Sanity: the hand-computed expectation for 1 unit held from bar 0.
    assert off_rewards == pytest.approx([0.0, 10.0, -5.0, 15.0, -5.0], abs=1e-9)


def test_offline_realtime_metric_parity():
    """Reward-stream-derived evaluate() metrics match across both tasks."""
    _, off_metrics = _run_offline(_TAPE, _BUY_THEN_HOLD)
    _, rt_metrics = _run_realtime(_TAPE, _BUY_THEN_HOLD)
    for key in _SHARED_METRIC_KEYS:
        assert key in off_metrics, f"offline missing {key}"
        assert key in rt_metrics, f"realtime missing {key}"
        assert off_metrics[key] == pytest.approx(rt_metrics[key], abs=1e-9), (
            f"metric {key} diverged: offline={off_metrics[key]} "
            f"realtime={rt_metrics[key]}"
        )


def test_offline_total_return_snapshot():
    """Offline headline total_return is the buy-and-hold sum (frozen).

    Guards the offline reward against silent drift through the refactor:
    1 unit held from bar 0 yields Σ(close[t]-close[t-1]) = 115-100 = 15.
    """
    _, off_metrics = _run_offline(_TAPE, _BUY_THEN_HOLD)
    assert off_metrics["total_return"] == pytest.approx(15.0, abs=1e-9)
