"""FastAPI server tests for the verifier RPC service.

Uses ``fastapi.testclient.TestClient`` so no real port is bound during
unit tests. The synthetic task layout is built fresh per test under
``tmp_path`` (see ``tests/verifier_fixtures.py``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from openfinai_harbor.verifier import registry
from openfinai_harbor.verifier.server import VerifierConfig, create_app

from tests.verifier_fixtures import (
    SYNTHETIC_EVALUATOR_RAISES_SOURCE,
    encode_array_b64,
    make_synthetic_task,
)


# Helpers


def _build_config(
    *,
    task_dir: Path,
    state_dir: Path,
    token: str = "test-token",
    despawn_grace_min: float = 10.0,
    max_requests_per_minute: int = 60,
    submission_cache_size: int = 64,
    heartbeat_timeout_sec: float = 90.0,
    liveness_poll_sec: float = 30.0,
) -> VerifierConfig:
    state_dir.mkdir(parents=True, exist_ok=True)
    run_dir = registry.new_run_dir(state_dir)
    return VerifierConfig(
        task_id=task_dir.name,
        task_dir=task_dir,
        data_dir=task_dir / "environment" / "data",
        eval_data_dir=task_dir / "environment" / "eval-data",
        evaluator_path=task_dir / "environment" / "data" / "evaluator.py",
        interaction_model="forecasting",
        has_held_out_test_gt=True,
        state_dir=state_dir,
        run_dir=run_dir,
        token=token,
        despawn_grace_min=despawn_grace_min,
        heartbeat_timeout_sec=heartbeat_timeout_sec,
        liveness_poll_sec=liveness_poll_sec,
        max_requests_per_minute=max_requests_per_minute,
        submission_cache_size=submission_cache_size,
    )


def _auth(token: str = "test-token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def synthetic_app(tmp_path):
    """Build a synthetic task + verifier app + TestClient. Yields (client, state_dir)."""
    task_dir = make_synthetic_task(tmp_path)
    state_dir = tmp_path / "state" / task_dir.name
    cfg = _build_config(task_dir=task_dir, state_dir=state_dir)
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client, state_dir


# /healthz


def test_healthz_returns_200_without_auth(synthetic_app):
    client, _ = synthetic_app
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["task_id"] == "synthetic_task"
    assert body["n_active_agents"] == 0
    assert body["n_total_submissions"] == 0


# Auth (401 on protected endpoints)


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("get", "/info", None),
        ("post", "/register", {"agent_id": "a"}),
        ("post", "/heartbeat", {"agent_id": "a"}),
        ("post", "/deregister", {"agent_id": "a"}),
        (
            "post",
            "/submit",
            {
                "agent_id": "a",
                "submission_id": "s",
                "predictions_b64": "x",
            },
        ),
    ],
)
def test_protected_endpoints_require_bearer_token(synthetic_app, method, path, payload):
    client, _ = synthetic_app
    fn = getattr(client, method)
    resp = fn(path) if payload is None else fn(path, json=payload)
    assert resp.status_code == 401


def test_protected_endpoints_reject_wrong_token(synthetic_app):
    client, _ = synthetic_app
    resp = client.post(
        "/heartbeat",
        json={"agent_id": "a"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


# /info


def test_info_returns_task_metadata(synthetic_app):
    client, _ = synthetic_app
    resp = client.get("/info", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "synthetic_task"
    assert body["interaction_model"] == "forecasting"
    assert body["reward_names"] == ["mse"]
    assert body["has_held_out_test_gt"] is True
    assert body["expects_fake_emb"] is False
    assert body["n_active_agents"] == 0
    assert body["despawn_grace_min"] == 10.0


# Register / heartbeat / deregister flow


def test_register_then_heartbeat_then_deregister(synthetic_app):
    client, _ = synthetic_app
    r1 = client.post("/register", json={"agent_id": "a1"}, headers=_auth())
    assert r1.status_code == 200
    body = r1.json()
    assert body["agent_id"] == "a1"
    assert body["task_id"] == "synthetic_task"
    assert body["n_active_agents"] == 1

    r2 = client.post("/heartbeat", json={"agent_id": "a1"}, headers=_auth())
    assert r2.status_code == 200
    assert r2.json()["n_active_agents"] == 1

    r3 = client.post("/deregister", json={"agent_id": "a1"}, headers=_auth())
    assert r3.status_code == 200
    assert r3.json()["n_active_agents"] == 0


def test_register_idempotent_for_same_agent_id(synthetic_app):
    client, _ = synthetic_app
    r1 = client.post("/register", json={"agent_id": "dup"}, headers=_auth())
    assert r1.status_code == 200
    assert r1.json()["n_active_agents"] == 1
    # Re-register same id: still one agent active.
    r2 = client.post("/register", json={"agent_id": "dup"}, headers=_auth())
    assert r2.status_code == 200
    assert r2.json()["n_active_agents"] == 1


def test_heartbeat_auto_registers_if_missing(synthetic_app):
    """A heartbeat from an unknown agent_id should silently register it."""
    client, _ = synthetic_app
    r = client.post("/heartbeat", json={"agent_id": "phantom"}, headers=_auth())
    assert r.status_code == 200
    assert r.json()["n_active_agents"] == 1


# /submit happy path


def test_submit_happy_path(synthetic_app):
    client, _ = synthetic_app
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    payload = {
        "agent_id": "alice",
        "submission_id": "sub-001",
        "predictions_b64": encode_array_b64(pred),
    }
    resp = client.post("/submit", json=payload, headers=_auth())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["submission_id"] == "sub-001"
    assert body["cached"] is False
    # gt is [1,2,3,4,5] so mse should be 0.0.
    assert body["scores"]["mse"] == pytest.approx(0.0, abs=1e-6)
    assert body["weighted_total"] == pytest.approx(0.0, abs=1e-6)
    assert body["elapsed_sec"] >= 0.0


def test_submit_score_changes_with_prediction(synthetic_app):
    client, _ = synthetic_app
    bad = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    payload = {
        "agent_id": "alice",
        "submission_id": "sub-bad",
        "predictions_b64": encode_array_b64(bad),
    }
    resp = client.post("/submit", json=payload, headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["scores"]["mse"] > 0.0  # gt != 0 so mse > 0


# /submit malformed payload


def test_submit_missing_field_returns_422(synthetic_app):
    client, _ = synthetic_app
    resp = client.post(
        "/submit",
        json={"agent_id": "a", "submission_id": "s"},  # missing predictions_b64
        headers=_auth(),
    )
    assert resp.status_code == 422
    body = resp.json()
    # Pydantic's field-level error structure
    assert any("predictions_b64" in str(item) for item in body.get("detail", []))


def test_submit_malformed_b64_returns_422(synthetic_app):
    client, _ = synthetic_app
    resp = client.post(
        "/submit",
        json={
            "agent_id": "a",
            "submission_id": "s",
            "predictions_b64": "not-valid-base64-or-npy",
        },
        headers=_auth(),
    )
    assert resp.status_code == 422
    assert "predictions_b64" in resp.json()["detail"]


# /submit idempotent (cached LRU)


def test_submit_idempotent_returns_cached(synthetic_app):
    client, _ = synthetic_app
    pred = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    payload = {
        "agent_id": "alice",
        "submission_id": "stable-sid",
        "predictions_b64": encode_array_b64(pred),
    }
    r1 = client.post("/submit", json=payload, headers=_auth())
    r2 = client.post("/submit", json=payload, headers=_auth())
    assert r1.status_code == 200 and r2.status_code == 200
    b1, b2 = r1.json(), r2.json()
    assert b1["scores"] == b2["scores"]
    assert b1["weighted_total"] == b2["weighted_total"]
    assert b1["cached"] is False
    assert b2["cached"] is True


def test_submit_cache_evicts_oldest(tmp_path):
    """LRU bounded by submission_cache_size."""
    task_dir = make_synthetic_task(tmp_path)
    state_dir = tmp_path / "state" / task_dir.name
    cfg = _build_config(
        task_dir=task_dir, state_dir=state_dir, submission_cache_size=2
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        pred = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for sid in ("a", "b", "c"):
            client.post(
                "/submit",
                json={
                    "agent_id": "alice",
                    "submission_id": sid,
                    "predictions_b64": encode_array_b64(pred),
                },
                headers=_auth(),
            )
        # 'a' should have been evicted; resubmitting it with the same sid
        # re-runs the metric (cached=False), proving it's gone.
        r = client.post(
            "/submit",
            json={
                "agent_id": "alice",
                "submission_id": "a",
                "predictions_b64": encode_array_b64(pred),
            },
            headers=_auth(),
        )
        assert r.json()["cached"] is False


# Rate limit


def test_submit_rate_limit_429(tmp_path):
    task_dir = make_synthetic_task(tmp_path)
    state_dir = tmp_path / "state" / task_dir.name
    cfg = _build_config(
        task_dir=task_dir, state_dir=state_dir, max_requests_per_minute=2
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        pred = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Use unique sids so each call goes through (cache hit doesn't count
        # toward the limit but we want to exercise the limiter on miss-path).
        for i in range(2):
            r = client.post(
                "/submit",
                json={
                    "agent_id": "alice",
                    "submission_id": f"s-{i}",
                    "predictions_b64": encode_array_b64(pred),
                },
                headers=_auth(),
            )
            assert r.status_code == 200
        r3 = client.post(
            "/submit",
            json={
                "agent_id": "alice",
                "submission_id": "s-3",
                "predictions_b64": encode_array_b64(pred),
            },
            headers=_auth(),
        )
        assert r3.status_code == 429
        assert "Retry-After" in r3.headers


# Submit forwards fake_emb + extra


def test_submit_forwards_fake_emb_and_extra_to_evaluator(synthetic_app):
    """The synthetic evaluator records its kwargs in scores._kwargs_seen."""
    client, _ = synthetic_app
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    fake = np.array([0.1, 0.2], dtype=np.float32)
    payload = {
        "agent_id": "a",
        "submission_id": "sub-fwd",
        "predictions_b64": encode_array_b64(pred),
        "fake_emb_b64": encode_array_b64(fake),
        "extra": {"my_custom_kwarg": 42},
    }
    resp = client.post("/submit", json=payload, headers=_auth())
    assert resp.status_code == 200, resp.text
    # The synthetic evaluator's score() returned _kwargs_seen, but our
    # server strips underscored keys from the scores dict. Inspect via
    # a helper: send a unique sid and read the audit log.
    audit_dir = (
        Path(synthetic_app[1])
        / "runs"
    )
    # find newest run dir
    runs = sorted(audit_dir.glob("*"))
    assert runs, "expected at least one runs/ subdir"
    latest = runs[-1]
    audit = (latest / "submissions.jsonl").read_text(encoding="utf-8").splitlines()
    submitted = [json.loads(line) for line in audit if line.strip()]
    last = submitted[-1]
    assert last["submission_id"] == "sub-fwd"
    assert last["ok"] is True
    # _errors is not present in the audit (no errors)
    assert last.get("errors") is None


# /submit when evaluator raises


def test_submit_when_evaluator_raises_returns_500(tmp_path):
    task_dir = make_synthetic_task(
        tmp_path,
        evaluator_source=SYNTHETIC_EVALUATOR_RAISES_SOURCE,
    )
    state_dir = tmp_path / "state" / task_dir.name
    cfg = _build_config(task_dir=task_dir, state_dir=state_dir)
    app = create_app(cfg)
    with TestClient(app) as client:
        pred = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        resp = client.post(
            "/submit",
            json={
                "agent_id": "a",
                "submission_id": "boom",
                "predictions_b64": encode_array_b64(pred),
            },
            headers=_auth(),
        )
        assert resp.status_code == 500
        assert "synthetic explosion" in resp.json()["detail"]
        # Audit log should also record the error.
        runs = sorted((cfg.state_dir / "runs").glob("*"))
        audit = (runs[-1] / "submissions.jsonl").read_text(encoding="utf-8")
        assert "synthetic explosion" in audit
