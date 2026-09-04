"""Integration tests for the realtime_polymarket curated handler.

Spins up the FastAPI app from the actual ``tasks/realtime_polymarket``
bundle with a stub PolymarketProvider injected so no network is hit.
Covers:

- Handler registration in FAMILY_HANDLERS / handler_for.
- ``GET /markets/active`` returns the discovered universe with all
  agent-facing fields.
- ``POST /submit/event_predictions_async`` writes per-prediction rows
  to the ledger with per-market ``resolve_at`` (NOT a uniform horizon).
- ``entry_price`` is captured server-side from the provider's
  ``get_current_price``.
- Validation: probability out-of-bounds (handled by pydantic), unknown
  symbol (handled by handler).
- Deferred sidecar metadata includes provider="polymarket".
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tasks"))

from fastapi.testclient import TestClient

from openfinai_harbor.verifier.curated.handlers import (
    FAMILY_HANDLERS,
    handler_for,
)
from openfinai_harbor.verifier.curated.server import (
    CuratedVerifierConfig,
    create_curated_app,
)
from openfinai_harbor.verifier.curated.task_loader import CuratedTaskSpec
from openfinai_pipeline.realtime.data_providers.base import MarketSnapshot


# Fixtures


def _stub_market(symbol: str, *, end_offset_hours: float = 24, current_price: float = 0.5):
    return {
        "symbol": symbol,
        "question": f"Will event {symbol} happen?",
        "description": "Test market.",
        "resolution_at": datetime.now(timezone.utc) + timedelta(hours=end_offset_hours),
        "current_price": current_price,
        "outcomes": ["Yes", "No"],
        "outcome_prices": [current_price, 1 - current_price],
        "clob_token_ids": [f"tok-{symbol}-y", f"tok-{symbol}-n"],
        "best_bid": current_price - 0.01,
        "best_ask": current_price + 0.01,
        "spread": 0.02,
        "volume_24h": 5000.0,
        "liquidity": 1000.0,
        "tags": ["test"],
        "categories": ["politics"],
        "slug": f"slug-{symbol}",
    }


@pytest.fixture
def stub_polymarket_provider(monkeypatch):
    """Patch PolymarketProvider so handler tests need no network."""
    from openfinai_pipeline.realtime.data_providers import polymarket as pm_mod

    markets = [
        _stub_market("0xa", end_offset_hours=2, current_price=0.6),
        _stub_market("0xb", end_offset_hours=24, current_price=0.4),
    ]

    def fake_init(self, *args, **kwargs):
        self._market_cache = {}
        self._clob_token_cache = {}
        self.name = "polymarket"

    def fake_discover(self, filters):
        for m in markets:
            self._market_cache[m["symbol"]] = m
        return list(markets)

    def fake_get_current_price(self, symbol):
        for m in markets:
            if m["symbol"] == symbol:
                return MarketSnapshot(
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    price=float(m["current_price"]),
                )
        raise ValueError(f"no market {symbol}")

    def fake_get_event_outcome(self, symbol):
        return None

    def fake_get_metadata(self, symbol):
        return self._market_cache.get(symbol, {})

    monkeypatch.setattr(pm_mod.PolymarketProvider, "__init__", fake_init)
    monkeypatch.setattr(
        pm_mod.PolymarketProvider, "discover_active_markets", fake_discover
    )
    monkeypatch.setattr(
        pm_mod.PolymarketProvider, "get_current_price", fake_get_current_price
    )
    monkeypatch.setattr(
        pm_mod.PolymarketProvider, "get_event_outcome", fake_get_event_outcome
    )
    monkeypatch.setattr(
        pm_mod.PolymarketProvider, "get_market_metadata", fake_get_metadata
    )
    return markets


@pytest.fixture
def state_dirs(tmp_path):
    state_dir = tmp_path / "state"
    run_dir = state_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return state_dir, run_dir


def _build_app(state_dir, run_dir, ledger_path):
    spec = CuratedTaskSpec(
        task_dir=(REPO / "tasks" / "realtime_polymarket").resolve(),
        task_id="realtime_polymarket",
        family="realtime_polymarket",
        class_name="RealtimePolymarket",
        default_config={
            "db_path": str(ledger_path),
            "headline_metric": "brier_score",
        },
        task_module_path=(
            REPO / "tasks" / "realtime_polymarket" / "task.py"
        ).resolve(),
        raw_toml={},
    )
    cfg = CuratedVerifierConfig(
        spec=spec,
        state_dir=state_dir,
        run_dir=run_dir,
        token="test-token",
        despawn_grace_min=0.0,
        max_requests_per_minute=240,
        submission_cache_size=64,
    )
    return create_curated_app(cfg)


# Tests


class TestHandlerRegistration:
    def test_family_in_registry(self):
        assert "realtime_polymarket" in FAMILY_HANDLERS

    def test_handler_for_returns_module(self):
        mod = handler_for("realtime_polymarket")
        assert mod is not None
        assert hasattr(mod, "register_routes")
        assert hasattr(mod, "on_startup")
        assert mod.HEADLINE_METRIC == "brier_score"
        assert "brier_score" in mod.REWARD_NAMES


class TestMarketsEndpoint:
    def test_returns_discovered_universe(
        self, stub_polymarket_provider, state_dirs, tmp_path
    ):
        state_dir, run_dir = state_dirs
        ledger_path = tmp_path / "predictions.db"
        app = _build_app(state_dir, run_dir, ledger_path)
        client = TestClient(app)
        resp = client.get(
            "/markets/active",
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 200
        markets = resp.json()
        assert len(markets) == 2
        symbols = {m["symbol"] for m in markets}
        assert symbols == {"0xa", "0xb"}
        # Each entry has the agent-facing fields documented in
        # instruction.md.
        m0 = markets[0]
        for key in (
            "symbol",
            "question",
            "description",
            "resolution_at",
            "current_yes_price",
            "best_bid",
            "best_ask",
            "spread",
            "volume_24h",
            "liquidity",
            "tags",
            "categories",
        ):
            assert key in m0


class TestSubmitEventPredictions:
    def test_predictions_ledgered_returns_deferred(
        self, stub_polymarket_provider, state_dirs, tmp_path
    ):
        state_dir, run_dir = state_dirs
        ledger_path = tmp_path / "predictions.db"
        app = _build_app(state_dir, run_dir, ledger_path)
        client = TestClient(app)

        resp = client.post(
            "/submit/event_predictions_async",
            json={
                "agent_id": "test",
                "submission_id": "ev-1",
                "predictions": [
                    {"symbol": "0xa", "predicted_yes_probability": 0.7},
                    {"symbol": "0xb", "predicted_yes_probability": 0.4},
                ],
                # Back-pointer for orphan detection downstream — must
                # round-trip into the ledger row.
                "trial_dir": "/host/trials/realtime_polymarket/20260514T000000Z",
            },
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "deferred"
        assert body["deferred"]["n_predictions"] == 2
        assert body["deferred"]["provider"] == "polymarket"
        assert body["deferred"]["resolve_at_first"]
        assert body["deferred"]["resolve_at_last"]

        # The handler overrides db_path to <run_dir>/predictions.db so
        # the ledger is co-located with the trial's other artifacts. The
        # explicit `ledger_path` we passed in default_config is ignored.
        actual_ledger_path = run_dir / "predictions.db"
        assert actual_ledger_path.exists(), (
            f"expected ledger at {actual_ledger_path} (override target); "
            f"explicit ledger_path was {ledger_path}"
        )
        # The deferred sidecar must report the overridden path so the
        # resolver can find it.
        assert body["deferred"]["ledger_path"] == str(actual_ledger_path)

        # Sanity-check ledger rows
        conn = sqlite3.connect(str(actual_ledger_path))
        rows = conn.execute(
            "SELECT symbol, predicted_price, entry_price, direction, "
            "       resolve_at, status, snapshot, trial_dir FROM predictions "
            "WHERE session_id=? ORDER BY symbol",
            (body["deferred"]["session_id"],),
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        # Probabilities stored in predicted_price (semantic re-use)
        assert rows[0][0] == "0xa"
        assert rows[0][1] == pytest.approx(0.7)
        # entry_price captured server-side from stub
        assert rows[0][2] == pytest.approx(0.6)
        assert rows[0][3] == "long"  # constant for events
        assert rows[0][5] == "pending"
        # 0xa resolves earlier (2h vs 24h offset) — verify per-market resolve_at
        assert rows[0][4] < rows[1][4]
        # Snapshot persists human-readable market context so inspection
        # tools can show what the agent was betting on without re-fetching
        # Polymarket. Verify the cache-hit metadata copy worked.
        snap = json.loads(rows[0][6])
        assert snap["event_prediction"] is True
        assert snap["predicted_yes_probability"] == pytest.approx(0.7)
        assert snap["question"] == "Will event 0xa happen?"
        assert snap["categories"] == ["politics"]
        assert snap["end_date_iso"] is not None  # ISO string from stub
        # trial_dir stamped from the submission body
        assert rows[0][7] == "/host/trials/realtime_polymarket/20260514T000000Z"

    def test_unknown_symbol_rejected(
        self, stub_polymarket_provider, state_dirs, tmp_path
    ):
        state_dir, run_dir = state_dirs
        ledger_path = tmp_path / "predictions.db"
        app = _build_app(state_dir, run_dir, ledger_path)
        client = TestClient(app)
        resp = client.post(
            "/submit/event_predictions_async",
            json={
                "agent_id": "test",
                "submission_id": "ev-bad",
                "predictions": [
                    {"symbol": "0xnope", "predicted_yes_probability": 0.5},
                ],
            },
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 422
        assert "not in" in resp.text.lower()

    def test_probability_above_one_rejected(
        self, stub_polymarket_provider, state_dirs, tmp_path
    ):
        state_dir, run_dir = state_dirs
        ledger_path = tmp_path / "predictions.db"
        app = _build_app(state_dir, run_dir, ledger_path)
        client = TestClient(app)
        resp = client.post(
            "/submit/event_predictions_async",
            json={
                "agent_id": "test",
                "submission_id": "ev-hi",
                "predictions": [
                    {"symbol": "0xa", "predicted_yes_probability": 1.5},
                ],
            },
            headers={"Authorization": "Bearer test-token"},
        )
        # Pydantic validation error → 422
        assert resp.status_code == 422

    def test_probability_below_zero_rejected(
        self, stub_polymarket_provider, state_dirs, tmp_path
    ):
        state_dir, run_dir = state_dirs
        ledger_path = tmp_path / "predictions.db"
        app = _build_app(state_dir, run_dir, ledger_path)
        client = TestClient(app)
        resp = client.post(
            "/submit/event_predictions_async",
            json={
                "agent_id": "test",
                "submission_id": "ev-lo",
                "predictions": [
                    {"symbol": "0xa", "predicted_yes_probability": -0.1},
                ],
            },
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 422
