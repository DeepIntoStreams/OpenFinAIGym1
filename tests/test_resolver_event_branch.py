"""Resolve and score event-shaped sessions without changing trading sessions."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from data.knowledge_base.rewards.reward_bank import (
    ALL_EVENT_REWARDS,
    EventBrierScore,
    EventLogLoss,
)
from openfinai_pipeline.realtime.data_providers.base import (
    DataProvider,
    EventDataProvider,
    MarketSnapshot,
)
from openfinai_pipeline.realtime.ledger import PredictionLedger
from openfinai_pipeline.realtime.resolver import ResolverService
from openfinai_pipeline.settings import ResolverConfig


# Mocks


class _MockEventProvider:
    """EventDataProvider returning canned outcomes per symbol."""

    name = "polymarket"

    def __init__(self, outcomes: Dict[str, float | None]):
        self._outcomes = outcomes

    # DataProvider
    def get_current_price(self, symbol: str) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            price=0.5,
        )

    def get_price_at(self, symbol, at, interval="1m"):
        return self.get_current_price(symbol)

    def get_bars(self, symbol, interval, start, end):
        return []

    # EventDataProvider
    def discover_active_markets(self, filters):
        return []

    def get_event_outcome(self, symbol: str) -> float | None:
        return self._outcomes.get(symbol)

    def get_market_metadata(self, symbol: str) -> dict:
        return {}


class _MockTradingProvider:
    """Continuous-asset provider returning a fixed exit price."""

    name = "binance"

    def __init__(self, exit_price: float = 110.0):
        self._exit_price = exit_price

    def get_current_price(self, symbol):
        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            price=100.0,
        )

    def get_price_at(self, symbol, at, interval="1m"):
        return MarketSnapshot(symbol=symbol, timestamp=at, price=self._exit_price)

    def get_bars(self, symbol, interval, start, end):
        return []


# Helpers


def _seed_event_row(
    ledger: PredictionLedger,
    *,
    session_id: str,
    symbol: str,
    predicted_yes_probability: float,
    submitted_at: datetime | None = None,
    resolve_at: datetime | None = None,
    entry_price: float = 0.5,
) -> str:
    """Insert a pending event-shape row mirroring the polymarket handler."""
    submitted_at = submitted_at or datetime.now(timezone.utc) - timedelta(hours=2)
    resolve_at = resolve_at or datetime.now(timezone.utc) - timedelta(minutes=10)
    return ledger.submit(
        {
            "session_id": session_id,
            "provider": "polymarket",
            "symbol": symbol,
            "horizon_minutes": 60,
            "submitted_at": submitted_at,
            "resolve_at": resolve_at,
            "direction": "long",
            "predicted_price": predicted_yes_probability,
            "entry_price": entry_price,
            "snapshot": {"event_prediction": True},
        }
    )


def _seed_trading_row(
    ledger: PredictionLedger,
    *,
    session_id: str,
    symbol: str,
    direction: str,
    submitted_at: datetime | None = None,
    resolve_at: datetime | None = None,
    entry_price: float = 100.0,
) -> str:
    submitted_at = submitted_at or datetime.now(timezone.utc) - timedelta(hours=2)
    resolve_at = resolve_at or datetime.now(timezone.utc) - timedelta(minutes=10)
    return ledger.submit(
        {
            "session_id": session_id,
            "provider": "binance",
            "symbol": symbol,
            "horizon_minutes": 5,
            "submitted_at": submitted_at,
            "resolve_at": resolve_at,
            "direction": direction,
            "predicted_price": None,
            "entry_price": entry_price,
            "snapshot": {},
        }
    )


@pytest.fixture
def ledger(tmp_path):
    db = tmp_path / "test_resolver.db"
    lg = PredictionLedger(str(db))
    yield lg
    lg.close()


# Event resolution


class TestResolvePendingEvent:
    def test_yes_outcome_marked_resolved(self, ledger, tmp_path):
        sid = "s1"
        pid = _seed_event_row(
            ledger, session_id=sid, symbol="0xa", predicted_yes_probability=0.7
        )
        provider = _MockEventProvider({"0xa": 1.0})
        resolver = ResolverService(
            ledger,
            {"polymarket": provider},
            reward_classes=[cls() for cls in ALL_EVENT_REWARDS],
        )
        stats = resolver.resolve_pending()
        assert stats["resolved"] == 1
        row = ledger.get_session(sid)[0]
        assert row["status"] == "resolved"
        gt = json.loads(row["ground_truth"])
        assert gt["outcome"] == 1.0
        # actual_return is NOT meaningful for events — should stay null.
        assert row["actual_return"] is None
        assert row["exit_price"] is None

    def test_no_outcome_marked_resolved(self, ledger):
        pid = _seed_event_row(
            ledger, session_id="s2", symbol="0xb", predicted_yes_probability=0.3
        )
        provider = _MockEventProvider({"0xb": 0.0})
        resolver = ResolverService(
            ledger, {"polymarket": provider},
            reward_classes=[cls() for cls in ALL_EVENT_REWARDS],
        )
        stats = resolver.resolve_pending()
        assert stats["resolved"] == 1

    def test_ambiguous_outcome_marked_resolved(self, ledger):
        """0.5 outcomes ARE resolved (so resolver doesn't keep polling),
        but they get dropped at the Phase 2 scoring step."""
        pid = _seed_event_row(
            ledger, session_id="s3", symbol="0xc", predicted_yes_probability=0.5
        )
        provider = _MockEventProvider({"0xc": 0.5})
        resolver = ResolverService(
            ledger, {"polymarket": provider},
            reward_classes=[cls() for cls in ALL_EVENT_REWARDS],
        )
        stats = resolver.resolve_pending()
        assert stats["resolved"] == 1
        row = ledger.get_session("s3")[0]
        assert row["status"] == "resolved"
        gt = json.loads(row["ground_truth"])
        assert gt["outcome"] == 0.5

    def test_pending_outcome_leaves_row_pending(self, ledger):
        """When the upstream market hasn't resolved yet, the row should
        STAY pending (not be marked failed)."""
        pid = _seed_event_row(
            ledger, session_id="s4", symbol="0xd", predicted_yes_probability=0.4
        )
        provider = _MockEventProvider({"0xd": None})
        resolver = ResolverService(
            ledger, {"polymarket": provider},
            reward_classes=[cls() for cls in ALL_EVENT_REWARDS],
        )
        stats = resolver.resolve_pending()
        assert stats["resolved"] == 0
        assert stats["failed"] == 0
        row = ledger.get_session("s4")[0]
        assert row["status"] == "pending"


# Event scoring


class TestScoreResolvedEvent:
    def _seed_and_resolve_event_session(
        self,
        ledger: PredictionLedger,
        session_id: str,
        rows: List[tuple[str, float, float]],  # (symbol, predicted_prob, outcome)
    ) -> None:
        outcomes: Dict[str, float | None] = {}
        for symbol, prob, outcome in rows:
            _seed_event_row(
                ledger,
                session_id=session_id,
                symbol=symbol,
                predicted_yes_probability=prob,
            )
            outcomes[symbol] = outcome
        provider = _MockEventProvider(outcomes)
        resolver = ResolverService(
            ledger, {"polymarket": provider},
            reward_classes=[cls() for cls in ALL_EVENT_REWARDS],
        )
        resolver.resolve_pending()

    def test_brier_score_hand_computed(self, ledger):
        self._seed_and_resolve_event_session(
            ledger,
            "s_brier",
            [
                ("0xa", 0.7, 1.0),  # (0.7-1)^2 = 0.09
                ("0xb", 0.3, 0.0),  # (0.3-0)^2 = 0.09
                ("0xc", 0.6, 1.0),  # (0.6-1)^2 = 0.16
            ],
        )
        resolver = ResolverService(
            ledger,
            {"polymarket": _MockEventProvider({})},
            reward_classes=[cls() for cls in ALL_EVENT_REWARDS],
        )
        stats = resolver.score_resolved()
        assert stats["scored"] == 3
        rows = ledger.get_session("s_brier")
        rewards = json.loads(rows[0]["rewards"])
        assert rewards["brier_score"] == pytest.approx((0.09 + 0.09 + 0.16) / 3)
        # log_loss and ECE also populated
        assert "log_loss" in rewards
        assert "expected_calibration_error" in rewards

    def test_ambiguous_outcomes_dropped_with_count(self, ledger):
        """0.5 outcomes should be dropped from scoring, but counted."""
        self._seed_and_resolve_event_session(
            ledger,
            "s_amb",
            [
                ("0xa", 0.7, 1.0),
                ("0xb", 0.5, 0.5),  # ambiguous → dropped
                ("0xc", 0.4, 0.0),
            ],
        )
        resolver = ResolverService(
            ledger,
            {"polymarket": _MockEventProvider({})},
            reward_classes=[cls() for cls in ALL_EVENT_REWARDS],
        )
        resolver.score_resolved()
        rows = ledger.get_session("s_amb")
        rewards = json.loads(rows[0]["rewards"])
        # Scored only the 2 non-ambiguous: (0.7-1)^2 + (0.4-0)^2 = 0.09 + 0.16 = 0.25 / 2
        assert rewards["brier_score"] == pytest.approx((0.09 + 0.16) / 2)
        assert rewards["n_dropped_ambiguous"] == 1.0

    def test_all_ambiguous_session_writes_nan_rewards(self, ledger):
        """Defensive: if EVERY outcome is 0.5 (pathological session), the
        rewards should be NaN-marked rather than 0.0-stamped, and the
        drop count should reflect everything."""
        self._seed_and_resolve_event_session(
            ledger,
            "s_all_amb",
            [
                ("0xa", 0.5, 0.5),
                ("0xb", 0.5, 0.5),
            ],
        )
        resolver = ResolverService(
            ledger,
            {"polymarket": _MockEventProvider({})},
            reward_classes=[cls() for cls in ALL_EVENT_REWARDS],
        )
        resolver.score_resolved()
        rows = ledger.get_session("s_all_amb")
        rewards = json.loads(rows[0]["rewards"])
        # Python's json round-trips NaN as a float NaN (non-standard JSON
        # but harmless for our internal SQLite payload — downstream
        # callers see math.isnan-detectable NaN).
        assert math.isnan(rewards["brier_score"])
        assert rewards["n_dropped_ambiguous"] == 2.0


# Trading compatibility


class TestTradingRegression:
    def test_trading_session_still_scores(self, ledger):
        sid = "trading_session"
        _seed_trading_row(
            ledger, session_id=sid, symbol="BTCUSDT", direction="long"
        )
        provider = _MockTradingProvider(exit_price=110.0)
        resolver = ResolverService(ledger, {"binance": provider})
        resolver.resolve_pending()
        stats = resolver.score_resolved()
        assert stats["scored"] == 1
        row = ledger.get_session(sid)[0]
        # actual_return = (110 - 100) / 100 = 0.1
        assert row["actual_return"] == pytest.approx(0.1)
        rewards = json.loads(row["rewards"])
        # direction=long, actual_return>0 → directional_accuracy = 1.0
        assert rewards["direction_accuracy"] == pytest.approx(1.0)
