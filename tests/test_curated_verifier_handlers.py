"""Deterministic FastAPI coverage for curated verifier handlers.

External providers are stubbed; tests requiring absent dataset caches skip.
"""

from __future__ import annotations

import base64
import io
import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Make tasks/ packages discoverable for import_task_class (which imports
# the module from a known absolute path, so this is more for sanity).
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
from openfinai_harbor.verifier.curated.task_loader import (
    CuratedTaskSpec,
    is_curated_bundle,
    load_curated_spec,
)


# Shared fixtures


@pytest.fixture
def state_dirs(tmp_path):
    """Return (state_dir, run_dir) under pytest's tmp_path."""
    state_dir = tmp_path / "state"
    run_dir = state_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return state_dir, run_dir


def _build_config(spec: CuratedTaskSpec, state_dir, run_dir):
    return CuratedVerifierConfig(
        spec=spec,
        state_dir=state_dir,
        run_dir=run_dir,
        token="test-token",
        despawn_grace_min=0.0,
        max_requests_per_minute=240,
        submission_cache_size=64,
    )


def _crypto_csv_present() -> bool:
    return (
        REPO
        / "data"
        / "pipeline_output"
        / "datasets"
        / "binance_crypto_hourly_ohlcv"
    ).exists() and any(
        (
            REPO
            / "data"
            / "pipeline_output"
            / "datasets"
            / "binance_crypto_hourly_ohlcv"
        ).glob("*.csv")
    )


# Family dispatch


class TestHandlerDispatch:
    def test_known_families_resolve(self):
        # realtime_trading deliberately omitted: those bundles run the
        # gym loop in the agent container (no host-side verifier role)
        # so they have no handler registered.
        for family in (
            "offline_forecasting",
            "offline_trading",
            "realtime_forecasting",
        ):
            mod = handler_for(family)
            assert mod is not None, f"family={family} unresolved"
            # Each handler must declare a HEADLINE_METRIC + REWARD_NAMES.
            assert hasattr(mod, "HEADLINE_METRIC"), family
            assert hasattr(mod, "REWARD_NAMES"), family
            assert hasattr(mod, "register_routes"), family

    def test_realtime_trading_intentionally_unhandled(self):
        # realtime_trading bundles run agent-side; the verifier is not
        # spawned for them. handler_for returns None.
        assert handler_for("realtime_trading") is None

    def test_unknown_family_returns_none(self):
        assert handler_for("nonexistent_family") is None

    def test_curated_bundles_with_verifier_have_handlers(self):
        # Trading bundles (offline + realtime) and forecasting bundles
        # all have task.toml + load_curated_spec works for them.
        # Only the families with host-side verifier roles need a handler.
        for bundle in (
            "offline_crypto_forecasting",
            "offline_stock_forecasting",
            "offline_crypto_trading",
            "offline_stock_trading",
            "realtime_crypto_forecasting",
            "realtime_stock_forecasting",
        ):
            task_dir = REPO / "tasks" / bundle
            assert is_curated_bundle(task_dir), bundle
            spec = load_curated_spec(task_dir)
            assert handler_for(spec.family) is not None, bundle

    def test_realtime_trading_bundles_load_but_have_no_handler(self):
        # Bundle metadata is still parseable; just no verifier handler.
        for bundle in (
            "realtime_crypto_trading",
            "realtime_stock_trading",
            "realtime_stock_trading_alpaca_paper",
        ):
            task_dir = REPO / "tasks" / bundle
            assert is_curated_bundle(task_dir), bundle
            spec = load_curated_spec(task_dir)
            assert spec.family == "realtime_trading"
            assert handler_for(spec.family) is None


# Forecasting handler


@pytest.mark.skipif(not _crypto_csv_present(), reason="no Binance CSV cache")
class TestForecastingHandler:
    def _build_app(self, state_dir, run_dir):
        spec = load_curated_spec(REPO / "tasks" / "offline_crypto_forecasting")
        cfg = _build_config(spec, state_dir, run_dir)
        return create_curated_app(cfg)

    def test_dataset_h5_persisted_with_per_symbol_layout(
        self, state_dirs
    ):
        state_dir, run_dir = state_dirs
        app = self._build_app(state_dir, run_dir)
        h5_path = (
            REPO / "tasks" / "offline_crypto_forecasting" / "environment"
            / "data" / "dataset.h5"
        )
        assert h5_path.exists()
        import h5py

        with h5py.File(h5_path, "r") as f:
            assert f.attrs["layout"] == "per_symbol"
            symbols = list(f.keys())
            assert "BTCUSDT" in symbols
            for ds in ("X_train", "y_train", "X_test"):
                assert f"BTCUSDT/{ds}" in f

    def test_perfect_predictions_score_one(self, state_dirs):
        state_dir, run_dir = state_dirs
        app = self._build_app(state_dir, run_dir)
        client = TestClient(app)
        state = app.state.state
        y_test = state.handler_state["forecasting"]["y_test"]

        buf = io.BytesIO()
        np.savez(buf, **{k: np.asarray(v) for k, v in y_test.items()})
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        resp = client.post(
            "/submit/predictions",
            json={
                "agent_id": "test",
                "submission_id": "perfect",
                "predictions_b64": b64,
                "predictions_format": "ndarray_dict",
            },
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Post-overhaul: perfect price predictions yield directional_accuracy=1.0
        # AND mape=0.0; the headline metric is `mape` per the toml default,
        # so weighted_total = mape = 0.0 here. Both assertions encode the
        # contract: perfect on directional, perfect on price-error.
        assert body["scores"]["directional_accuracy"] == 1.0
        assert body["scores"]["mape"] == pytest.approx(0.0, abs=1e-6)
        # weighted_total mirrors the configured headline metric
        # (mape default → 0.0 for perfect predictions).
        assert body["weighted_total"] == pytest.approx(0.0, abs=1e-6)


# Interactive offline trading


_HDR = {"Authorization": "Bearer test-token"}


def _run_interactive_session(client, actions, *, agent_id="sess", session_id="s1"):
    """Drive a full /session/reset -> /session/step* episode.

    Returns the terminal step response (carrying scores), or None if the
    episode never started.
    """
    r = client.post(
        "/session/reset",
        json={"agent_id": agent_id, "session_id": session_id},
        headers=_HDR,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] == session_id
    assert body["step"] == 0
    done = body["done"]
    final = None
    for action in actions:
        if done:
            break
        r = client.post(
            "/session/step",
            json={
                "agent_id": agent_id,
                "session_id": session_id,
                "action": action,
            },
            headers=_HDR,
        )
        assert r.status_code == 200, r.text
        final = r.json()
        done = final["done"]
    return final


@pytest.mark.skipif(not _crypto_csv_present(), reason="no Binance CSV cache")
class TestTradingInteractiveSession:
    """The verifier serves causal observations and scores server-side."""

    def _build_app(self, state_dir, run_dir, episode_length=50):
        spec = load_curated_spec(REPO / "tasks" / "offline_crypto_trading")
        spec.default_config["episode_length"] = episode_length
        cfg = _build_config(spec, state_dir, run_dir)
        return create_curated_app(cfg)

    def test_reset_returns_initial_observation(self, state_dirs):
        state_dir, run_dir = state_dirs
        client = TestClient(self._build_app(state_dir, run_dir))
        r = client.post(
            "/session/reset",
            json={"agent_id": "a", "session_id": "s1"},
            headers=_HDR,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["session_id"] == "s1"
        assert body["step"] == 0
        assert body["done"] is False
        obs = body["observation"]
        assert obs is not None
        assert "symbols" in obs and "BTCUSDT" in obs["symbols"]
        assert "portfolio" in obs

    def test_reset_mints_session_id_when_absent(self, state_dirs):
        state_dir, run_dir = state_dirs
        client = TestClient(self._build_app(state_dir, run_dir))
        r = client.post("/session/reset", json={"agent_id": "a"}, headers=_HDR)
        assert r.status_code == 200, r.text
        assert r.json()["session_id"]  # non-empty minted id

    def test_step_runs_to_done_and_scores_server_side(self, state_dirs):
        state_dir, run_dir = state_dirs
        client = TestClient(self._build_app(state_dir, run_dir))
        actions = [{"action": "hold"} for _ in range(60)]  # over-long; stops at done
        final = _run_interactive_session(client, actions)
        assert final is not None
        assert final["done"] is True
        assert "pnl" in final["scores"]
        assert final["weighted_total"] == final["scores"]["pnl"]
        # Flat policy -> ~zero PnL (matches the batch flat test).
        assert abs(final["scores"]["pnl"]) < 1e-6

    def test_long_short_pnl_mirror(self, state_dirs):
        # Buy-and-hold vs short-and-hold over the same window give
        # sign-mirrored PnL — a correctness check on the interactive
        # scoring path. Enter a small fractional position (well under the
        # 1x buying-power cap at any plausible BTC price) on the first bar,
        # then hold for the rest of the episode.
        state_dir, run_dir = state_dirs
        client = TestClient(self._build_app(state_dir, run_dir))
        long_open = {"action": "buy", "symbol": "BTCUSDT", "quantity": 0.1}
        short_open = {"action": "sell", "symbol": "BTCUSDT", "quantity": 0.1}
        hold = {"action": "hold"}
        longs = _run_interactive_session(
            client, [long_open] + [hold] * 59, agent_id="L", session_id="L1"
        )
        shorts = _run_interactive_session(
            client, [short_open] + [hold] * 59, agent_id="S", session_id="S1"
        )
        long_pnl = longs["scores"]["pnl"]
        short_pnl = shorts["scores"]["pnl"]
        assert long_pnl != 0  # sanity: real bars move
        assert abs(long_pnl + short_pnl) < abs(long_pnl) * 0.05

    def test_batch_endpoint_removed(self, state_dirs):
        # The foresight-leaky batch path is gone; only the interactive
        # session remains.
        state_dir, run_dir = state_dirs
        client = TestClient(self._build_app(state_dir, run_dir))
        r = client.post(
            "/submit/session",
            json={
                "agent_id": "a",
                "submission_id": "x",
                "actions": [{"action": "hold"}],
            },
            headers=_HDR,
        )
        assert r.status_code in (404, 405)

    def test_observation_advances_per_step(self, state_dirs):
        # Each step reveals the next bar only; the observed close changes as
        # the cursor advances (the agent cannot see future bars up front).
        state_dir, run_dir = state_dirs
        client = TestClient(self._build_app(state_dir, run_dir))
        r = client.post(
            "/session/reset",
            json={"agent_id": "a", "session_id": "s1"},
            headers=_HDR,
        )
        close0 = r.json()["observation"]["symbols"]["BTCUSDT"]["close"]
        r = client.post(
            "/session/step",
            json={"agent_id": "a", "session_id": "s1", "action": {"action": "hold"}},
            headers=_HDR,
        )
        body = r.json()
        assert body["step"] == 1
        assert body["observation"] is not None
        close1 = body["observation"]["symbols"]["BTCUSDT"]["close"]
        # Real hourly bars: consecutive closes differ.
        assert close0 != close1

    def test_unknown_session_id_404(self, state_dirs):
        state_dir, run_dir = state_dirs
        client = TestClient(self._build_app(state_dir, run_dir))
        r = client.post(
            "/session/step",
            json={"agent_id": "a", "session_id": "nope", "action": {"action": "hold"}},
            headers=_HDR,
        )
        assert r.status_code == 404

    def test_agent_mismatch_403(self, state_dirs):
        state_dir, run_dir = state_dirs
        client = TestClient(self._build_app(state_dir, run_dir))
        client.post(
            "/session/reset",
            json={"agent_id": "owner", "session_id": "s1"},
            headers=_HDR,
        )
        r = client.post(
            "/session/step",
            json={"agent_id": "intruder", "session_id": "s1", "action": {"action": "hold"}},
            headers=_HDR,
        )
        assert r.status_code == 403

    def test_step_after_done_409(self, state_dirs):
        state_dir, run_dir = state_dirs
        client = TestClient(self._build_app(state_dir, run_dir, episode_length=5))
        actions = [{"action": "hold"} for _ in range(5)]
        final = _run_interactive_session(client, actions)
        assert final["done"] is True
        r = client.post(
            "/session/step",
            json={"agent_id": "sess", "session_id": "s1", "action": {"action": "hold"}},
            headers=_HDR,
        )
        assert r.status_code == 409


# Realtime forecasting handler


class TestRealtimeForecastingHandler:
    """Realtime forecasting + deferred ledger.

    The handler captures entry_price server-side via
    ``provider.get_current_price()`` at submission time. We monkey-patch
    BinanceProvider so the test doesn't depend on Binance reachability
    and the ledgered entry price is deterministic.
    """

    _MOCK_ENTRY_PRICE = 65042.5

    @pytest.fixture(autouse=True)
    def _stub_binance(self, monkeypatch):
        from openfinai_pipeline.realtime.data_providers import (
            base as _base,
            binance as _binance,
        )

        def fake_get_current_price(self, symbol):
            return _base.MarketSnapshot(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                price=self.__class__._MOCK_PRICE,
                volume=1.0,
            )

        monkeypatch.setattr(
            _binance.BinanceProvider, "_MOCK_PRICE", self._MOCK_ENTRY_PRICE,
            raising=False,
        )
        monkeypatch.setattr(
            _binance.BinanceProvider,
            "get_current_price",
            fake_get_current_price,
        )

    def _build_app(self, state_dir, run_dir, ledger_path):
        spec = CuratedTaskSpec(
            task_dir=(REPO / "tasks" / "realtime_crypto_forecasting").resolve(),
            task_id="realtime_crypto_forecasting_btcusdt",
            family="realtime_forecasting",
            class_name="RealtimeCryptoForecasting",
            default_config={
                "symbols": ["BTCUSDT"],
                "horizon_bars": 10,
                "context_bars": 60,
                "db_path": str(ledger_path),
            },
            task_module_path=(
                REPO / "tasks" / "realtime_crypto_forecasting" / "task.py"
            ).resolve(),
            raw_toml={},
        )
        cfg = _build_config(spec, state_dir, run_dir)
        return create_curated_app(cfg)

    def test_predictions_ledgered_returns_deferred(self, tmp_path, state_dirs):
        state_dir, run_dir = state_dirs
        ledger_path = tmp_path / "predictions.db"
        app = self._build_app(state_dir, run_dir, ledger_path)
        client = TestClient(app)

        # Note: no entry_price field — the handler captures it
        # server-side from the (mocked) provider.
        predictions = [
            {
                "symbol": "BTCUSDT",
                "direction": "long",
            },
            {
                "symbol": "BTCUSDT",
                "direction": "short",
            },
        ]
        resp = client.post(
            "/submit/predictions_async",
            json={
                "agent_id": "test",
                "submission_id": "defer-1",
                "predictions": predictions,
                # Back-pointer to trial dir — round-trips into the ledger
                # row for orphan detection.
                "trial_dir": "/host/trials/realtime_crypto_forecasting/20260514T000000Z",
            },
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "deferred"
        assert body["deferred"]["n_predictions"] == 2
        assert body["deferred"]["session_id"]
        assert body["deferred"]["ledger_path"]

        # Verify rows exist in the SQLite ledger and that entry_price
        # was captured server-side from the mocked provider.
        # Handler overrides db_path to <run_dir>/predictions.db; explicit
        # ledger_path in default_config is ignored.
        conn = sqlite3.connect(str(run_dir / "predictions.db"))
        rows = conn.execute(
            "SELECT direction, entry_price, status, trial_dir FROM predictions "
            "WHERE session_id=?",
            (body["deferred"]["session_id"],),
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        assert all(r[2] == "pending" for r in rows)
        # Both rows should have the mocked entry price — the handler
        # ignored any agent attempt to set it (the field doesn't exist
        # on the wire model anymore).
        assert all(abs(r[1] - self._MOCK_ENTRY_PRICE) < 1e-6 for r in rows)
        # trial_dir stamped on every row from the submission body.
        expected_td = "/host/trials/realtime_crypto_forecasting/20260514T000000Z"
        assert all(r[3] == expected_td for r in rows)

    def test_agent_supplied_entry_price_is_ignored(
        self, tmp_path, state_dirs
    ):
        """Smuggling entry_price via extra is silently dropped — the
        verifier always captures the entry price server-side."""
        state_dir, run_dir = state_dirs
        ledger_path = tmp_path / "predictions.db"
        app = self._build_app(state_dir, run_dir, ledger_path)
        client = TestClient(app)

        resp = client.post(
            "/submit/predictions_async",
            json={
                "agent_id": "test",
                "submission_id": "fake-entry",
                "predictions": [
                    {
                        "symbol": "BTCUSDT",
                        "direction": "long",
                        "entry_price": 1.0,  # try to fabricate
                    },
                ],
            },
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 200
        # Handler overrides db_path to <run_dir>/predictions.db; explicit
        # ledger_path in default_config is ignored.
        conn = sqlite3.connect(str(run_dir / "predictions.db"))
        rows = conn.execute(
            "SELECT entry_price FROM predictions WHERE session_id=?",
            (resp.json()["deferred"]["session_id"],),
        ).fetchall()
        conn.close()
        # Verifier-fetched price wins, not the fabricated 1.0
        assert abs(rows[0][0] - self._MOCK_ENTRY_PRICE) < 1e-6

    def test_wrong_symbol_rejected(self, tmp_path, state_dirs):
        state_dir, run_dir = state_dirs
        ledger_path = tmp_path / "predictions.db"
        app = self._build_app(state_dir, run_dir, ledger_path)
        client = TestClient(app)

        resp = client.post(
            "/submit/predictions_async",
            json={
                "agent_id": "test",
                "submission_id": "bad-sym",
                "predictions": [
                    {"symbol": "XRPUSDT", "direction": "long"}
                ],
            },
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 422


# resolve_deferred CLI


class _StubProvider:
    """Deterministic provider stub for resolve_deferred end-to-end."""

    name = "binance"

    def __init__(self, exit_prices):
        self._exit_prices = exit_prices

    def get_current_price(self, symbol: str):
        from openfinai_pipeline.realtime.data_providers.base import MarketSnapshot

        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            price=self._exit_prices.get(symbol, 0.0),
            volume=1.0,
        )

    def get_price_at(self, symbol: str, when, interval: str = "1m"):
        from openfinai_pipeline.realtime.data_providers.base import MarketSnapshot

        return MarketSnapshot(
            symbol=symbol,
            timestamp=when,
            price=self._exit_prices.get(symbol, 0.0),
            volume=1.0,
        )


class TestResolveDeferred:
    def test_deferred_session_resolves_to_directional_accuracy(
        self, tmp_path, monkeypatch
    ):
        # 0. Stub BinanceProvider.get_current_price so the verifier-side
        #    entry-price capture is deterministic (and offline).
        from openfinai_pipeline.realtime.data_providers import (
            base as _base,
            binance as _binance,
        )
        entry_price = 65000.0

        def _fake_entry(self, symbol):
            return _base.MarketSnapshot(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                price=entry_price,
                volume=1.0,
            )

        monkeypatch.setattr(
            _binance.BinanceProvider, "get_current_price", _fake_entry
        )

        # 1. Submit deferred predictions to a fresh ledger.
        ledger_path = tmp_path / "predictions.db"
        spec = CuratedTaskSpec(
            task_dir=(REPO / "tasks" / "realtime_crypto_forecasting").resolve(),
            task_id="realtime_crypto_forecasting_btcusdt",
            family="realtime_forecasting",
            class_name="RealtimeCryptoForecasting",
            default_config={
                "symbols": ["BTCUSDT"],
                "horizon_bars": 1,
                "context_bars": 60,
                "db_path": str(ledger_path),
            },
            task_module_path=(
                REPO / "tasks" / "realtime_crypto_forecasting" / "task.py"
            ).resolve(),
            raw_toml={},
        )
        cfg = _build_config(spec, tmp_path / "state", tmp_path / "run")
        (tmp_path / "run").mkdir(parents=True, exist_ok=True)
        app = create_curated_app(cfg)
        client = TestClient(app)

        # 3 long + 1 short; exit > entry so 3/4 directional accuracy.
        # entry_price is captured server-side from the stubbed
        # BinanceProvider — agent doesn't supply it (the field is gone
        # from the wire model). submitted_at is backdated so the
        # bar-aligned resolve_at lands in the past and the resolver
        # processes immediately (horizon_bars >= 1 forbids the old
        # negative-horizon shortcut).
        past = (
            datetime.now(timezone.utc) - timedelta(minutes=10)
        ).isoformat()
        predictions = [
            {"symbol": "BTCUSDT", "direction": "long", "submitted_at": past},
            {"symbol": "BTCUSDT", "direction": "long", "submitted_at": past},
            {"symbol": "BTCUSDT", "direction": "long", "submitted_at": past},
            {"symbol": "BTCUSDT", "direction": "short", "submitted_at": past},
        ]
        resp = client.post(
            "/submit/predictions_async",
            json={
                "agent_id": "test",
                "submission_id": "resolve-test",
                "predictions": predictions,
            },
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 200
        deferred = resp.json()["deferred"]

        # 2. Lay out a trial-dir-shaped fixture: verifier/{reward.json,
        #    deferred_session.json}.
        trial_dir = tmp_path / "trial"
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True)
        (verifier_dir / "reward.json").write_text(
            json.dumps({"reward": 0.0, "status_deferred": 1.0}),
            encoding="utf-8",
        )
        (verifier_dir / "deferred_session.json").write_text(
            json.dumps(deferred),
            encoding="utf-8",
        )

        # 3. Patch resolve_deferred._build_provider so resolution doesn't
        #    hit the network. Exit price 65500 → all longs win, short loses.
        import openfinai_harbor.resolve_deferred as rd

        monkeypatch.setattr(
            rd, "_build_provider",
            lambda name: _StubProvider({"BTCUSDT": 65500.0}),
        )

        # 4. Run the CLI.
        exit_code = rd.main([str(trial_dir)])
        assert exit_code == 0

        # 5. reward.json (single-key) is overwritten with the headline
        #    metric; the full per-channel payload moved to metrics.json
        #    (see resolve_deferred._write_resolved_reward — schema split
        #    keeps harbor's Mean.compute happy on the single-key file).
        final = json.loads(
            (verifier_dir / "reward.json").read_text(encoding="utf-8")
        )
        assert final == {"reward": pytest.approx(0.75, abs=1e-6)}
        metrics = json.loads(
            (verifier_dir / "metrics.json").read_text(encoding="utf-8")
        )
        assert abs(metrics["directional_accuracy"] - 0.75) < 1e-6
        assert abs(metrics["reward"] - 0.75) < 1e-6
        assert metrics["status_resolved"] == 1.0
        assert metrics["status_deferred"] == 0.0
        assert metrics["n_resolved"] == 4.0
        assert metrics["n_total"] == 4.0

    def test_missing_sidecar_returns_exit_1(self, tmp_path):
        import openfinai_harbor.resolve_deferred as rd

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        # No sidecar — resolve_deferred should bail with exit 1.
        exit_code = rd.main([str(empty_dir)])
        assert exit_code == 1
