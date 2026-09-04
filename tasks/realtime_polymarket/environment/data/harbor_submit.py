"""In-container submit helper for curated tasks.

This file is bundled into curated agent bundles at install time
(``tasks/<task>/environment/data/harbor_submit.py``, mounted at
``/data/harbor_submit.py`` inside the container). It is
**deliberately self-contained** — only stdlib + ``requests`` + ``numpy``
— so the agent image doesn't need FastAPI / Pydantic.

Per-family submit functions:

* :func:`submit` — offline forecasting. Auto-detects ndarray
  vs dict-of-arrays and posts to ``/submit/predictions`` with the right
  ``predictions_format``.
* :func:`submit_and_persist` — like the above plus writes
  ``reward.json`` to ``HARBOR_REWARD_OUTPUT`` for harbor's verifier
  step to pick up.

Trading + realtime helpers are added by Phases B2/B3/B4.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, Optional

# numpy is *imported inside* the helpers rather than at module top so this
# file can be linted by hosts that don't have numpy. Inside an agent
# container numpy is always present.


_DEFAULT_TIMEOUT_SEC = 60.0
_INITIAL_RETRY_BUDGET_SEC = 10.0
_INITIAL_RETRY_BACKOFF_SEC = 0.5


# Errors


class VerifierUnreachableError(RuntimeError):
    pass


class VerifierAuthError(RuntimeError):
    pass


class VerifierRateLimitedError(RuntimeError):
    def __init__(self, message: str, retry_after_sec: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after_sec = retry_after_sec


class VerifierScoringError(RuntimeError):
    pass


# Encoding (auto-detect ndarray vs dict-of-arrays)


def _encode_predictions(predictions: Any) -> tuple[str, str]:
    """Encode ``predictions`` and return (b64_payload, format_tag).

    A plain array goes through ``np.save`` (auto-pipe wire shape, format
    'ndarray'). A dict goes through ``np.savez`` (one zip member per
    key, format 'ndarray_dict') — this is how multi-symbol curated
    forecasting tasks emit their predictions.
    """
    import numpy as np  # local import: see module docstring

    buf = io.BytesIO()
    if isinstance(predictions, dict):
        # np.savez accepts only ndarray-like values; coerce defensively
        # so a caller passing a dict-of-lists doesn't get a cryptic error.
        coerced = {str(k): np.asarray(v) for k, v in predictions.items()}
        np.savez(buf, **coerced)
        return base64.b64encode(buf.getvalue()).decode("ascii"), "ndarray_dict"
    np.save(buf, np.asarray(predictions), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "ndarray"


# Endpoint discovery


def _resolve_endpoint() -> tuple[str, str, str]:
    """Read VERIFIER_URL + VERIFIER_TOKEN + AGENT_ID from the environment."""
    url = os.environ.get("VERIFIER_URL", "").strip()
    token = os.environ.get("VERIFIER_TOKEN", "").strip()
    if not url or not token:
        raise RuntimeError(
            "VERIFIER_URL and VERIFIER_TOKEN must be set in the agent's "
            "environment (set by `docker run -e ...` or compose env)."
        )
    agent_id = os.environ.get("AGENT_ID", "").strip() or str(uuid.uuid4())
    return url, token, agent_id


# HTTP plumbing


def _do_post(
    url: str,
    token: str,
    payload: Dict[str, Any],
    *,
    timeout_sec: float,
    initial_retry_budget_sec: float = 0.0,
) -> Dict[str, Any]:
    import requests

    deadline = time.time() + max(initial_retry_budget_sec, 0.0)
    backoff = _INITIAL_RETRY_BACKOFF_SEC
    while True:
        try:
            resp = requests.post(
                url,
                data=json.dumps(payload),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=timeout_sec,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            if time.time() >= deadline:
                raise VerifierUnreachableError(
                    f"verifier unreachable at {url}: {exc}"
                ) from exc
            time.sleep(min(backoff, max(0.1, deadline - time.time())))
            backoff = min(backoff * 1.5, 2.0)
            continue
        return _parse_response(resp)


def _parse_response(resp: Any) -> Dict[str, Any]:
    if resp.status_code == 401:
        raise VerifierAuthError("verifier rejected token (401)")
    if resp.status_code == 429:
        retry_after_str = resp.headers.get("Retry-After", "0")
        try:
            retry_after = float(retry_after_str)
        except ValueError:
            retry_after = 0.0
        raise VerifierRateLimitedError(
            f"verifier rate-limited (429); retry-after={retry_after}s",
            retry_after,
        )
    if resp.status_code == 500:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise VerifierScoringError(f"verifier scoring error: {detail}")
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"verifier {resp.status_code}: {detail}")
    try:
        return resp.json()
    except ValueError as exc:
        raise RuntimeError(f"non-JSON response from verifier: {resp.text!r}") from exc


# Forecasting


def submit(
    predictions: Any,
    *,
    submission_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    initial_retry_budget_sec: float = _INITIAL_RETRY_BUDGET_SEC,
) -> Dict[str, Any]:
    """Submit ``predictions`` to the curated forecasting verifier.

    ``predictions`` can be either:

    * ``np.ndarray`` (single-output forecasting) — encoded via ``np.save``.
    * ``dict[str, np.ndarray]`` (multi-symbol forecasting) — encoded via
      ``np.savez``. Keys are typically symbol names.

    The wire format is auto-selected. Returns the parsed response dict
    with keys ``status``, ``scores``, ``weighted_total``, ``elapsed_sec``,
    ``submission_id``, ``cached``.
    """
    url, token, agent_id = _resolve_endpoint()
    sid = submission_id or str(uuid.uuid4())
    b64, fmt = _encode_predictions(predictions)
    payload: Dict[str, Any] = {
        "agent_id": agent_id,
        "submission_id": sid,
        "predictions_b64": b64,
        "predictions_format": fmt,
    }
    if extra:
        payload["extra"] = dict(extra)
    return _do_post(
        url.rstrip("/") + "/submit/predictions",
        token,
        payload,
        timeout_sec=timeout_sec,
        initial_retry_budget_sec=initial_retry_budget_sec,
    )


def submit_and_persist(
    predictions: Any,
    *,
    reward_output_path: Optional[str] = None,
    submission_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    initial_retry_budget_sec: float = _INITIAL_RETRY_BUDGET_SEC,
) -> Dict[str, Any]:
    """Submit predictions and persist verifier-compatible reward artifacts."""
    result = submit(
        predictions,
        submission_id=submission_id,
        extra=extra,
        timeout_sec=timeout_sec,
        initial_retry_budget_sec=initial_retry_budget_sec,
    )
    _persist_reward_json(result, reward_output_path)
    return result


# Trading


def submit_actions(
    actions: Any,
    *,
    submission_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    initial_retry_budget_sec: float = _INITIAL_RETRY_BUDGET_SEC,
) -> Dict[str, Any]:
    """Submit a trading action stream to the curated verifier.

    Used by offline trading bundles (``OfflineCryptoTrading`` /
    ``OfflineStockTrading``). Each action dict is a per-symbol position
    target::

        [{"BTCUSDT": 1, "ETHUSDT": -1}, ...]

    The verifier replays the stream against cached OHLCV, computes
    ``pnl`` / ``sharpe_ratio`` / ``max_drawdown`` / ``win_rate`` + scalar
    stats, and returns the aggregate response.

    Realtime trading bundles do NOT use this helper: they run the gym
    loop in the agent container against the live provider and write
    reward.json directly. The ``snapshots`` / ``realized_returns``
    parameters this helper used to take are gone — they were the
    fake-price exploit vector for realtime replay.
    """
    url, token, agent_id = _resolve_endpoint()
    sid = submission_id or str(uuid.uuid4())
    payload: Dict[str, Any] = {
        "agent_id": agent_id,
        "submission_id": sid,
        "actions": list(actions),
    }
    if extra:
        payload["extra"] = dict(extra)
    return _do_post(
        url.rstrip("/") + "/submit/session",
        token,
        payload,
        timeout_sec=timeout_sec,
        initial_retry_budget_sec=initial_retry_budget_sec,
    )


def submit_actions_and_persist(
    actions: Any,
    *,
    reward_output_path: Optional[str] = None,
    submission_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    initial_retry_budget_sec: float = _INITIAL_RETRY_BUDGET_SEC,
) -> Dict[str, Any]:
    """submit_actions + write reward.json to HARBOR_REWARD_OUTPUT.

    Same persistence semantics as ``submit_and_persist``:
    numeric-only payload that satisfies harbor's pydantic schema.
    """
    result = submit_actions(
        actions,
        submission_id=submission_id,
        extra=extra,
        timeout_sec=timeout_sec,
        initial_retry_budget_sec=initial_retry_budget_sec,
    )
    _persist_reward_json(result, reward_output_path)
    return result


# Deferred realtime forecasting


def submit_predictions_async(
    predictions: list,
    *,
    submission_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    initial_retry_budget_sec: float = _INITIAL_RETRY_BUDGET_SEC,
) -> Dict[str, Any]:
    """Submit deferred price/direction predictions to the verifier."""
    url, token, agent_id = _resolve_endpoint()
    sid = submission_id or str(uuid.uuid4())
    payload: Dict[str, Any] = {
        "agent_id": agent_id,
        "submission_id": sid,
        "predictions": list(predictions),
    }
    # Link ledger rows to their Harbor trial for orphan cleanup.
    trial_dir = os.environ.get("HARBOR_TRIAL_DIR")
    if trial_dir:
        payload["trial_dir"] = trial_dir
    if extra:
        payload["extra"] = dict(extra)
    return _do_post(
        url.rstrip("/") + "/submit/predictions_async",
        token,
        payload,
        timeout_sec=timeout_sec,
        initial_retry_budget_sec=initial_retry_budget_sec,
    )


def submit_predictions_async_and_persist(
    predictions: list,
    *,
    reward_output_path: Optional[str] = None,
    submission_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    initial_retry_budget_sec: float = _INITIAL_RETRY_BUDGET_SEC,
) -> Dict[str, Any]:
    """Submit deferred predictions and persist reward and resolver sidecars."""
    result = submit_predictions_async(
        predictions,
        submission_id=submission_id,
        extra=extra,
        timeout_sec=timeout_sec,
        initial_retry_budget_sec=initial_retry_budget_sec,
    )
    _persist_reward_json(result, reward_output_path)
    _persist_deferred_sidecar(result, reward_output_path)
    return result


def _persist_deferred_sidecar(
    result: Dict[str, Any], reward_output_path: Optional[str]
) -> None:
    """Write ``deferred_session.json`` next to reward.json.

    The sidecar lives in the same directory as reward.json so harbor's
    verifier output mount captures both. ``resolve_deferred`` looks for
    it under ``trial_dir/verifier/`` first, then ``trial_dir/`` as a
    fallback.
    """
    if result.get("status") != "deferred":
        return
    deferred_meta = result.get("deferred")
    if not isinstance(deferred_meta, dict) or not deferred_meta:
        return

    out_path = (
        reward_output_path
        or os.environ.get("HARBOR_REWARD_OUTPUT")
        or "/logs/verifier/reward.json"
    )
    try:
        from pathlib import Path as _Path

        sidecar_path = _Path(out_path).parent / "deferred_session.json"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(deferred_meta, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        sys.stderr.write(
            f"[harbor_submit] failed to persist deferred sidecar: "
            f"{exc}\n"
        )


# Shared persistence helper


def _persist_reward_json(
    result: Dict[str, Any], reward_output_path: Optional[str]
) -> None:
    """Write numeric scores, headline reward, and status flags."""
    out_path = (
        reward_output_path
        or os.environ.get("HARBOR_REWARD_OUTPUT")
        or "/logs/verifier/reward.json"
    )

    payload: Dict[str, float] = {}
    scores = result.get("scores") or {}
    if isinstance(scores, dict):
        for k, v in scores.items():
            if not isinstance(v, bool) and isinstance(v, (int, float)):
                payload[k] = v
    weighted_total = result.get("weighted_total", 0.0)
    if not isinstance(weighted_total, bool) and isinstance(
        weighted_total, (int, float)
    ):
        payload["reward"] = weighted_total
    else:
        payload["reward"] = 0.0

    sentinel_status = result.get("status")
    if sentinel_status == "scored":
        payload["status_scored"] = 1.0
    elif sentinel_status == "deferred":
        payload["status_deferred"] = 1.0

    try:
        from pathlib import Path as _Path

        p = _Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        sys.stderr.write(
            f"[harbor_submit] failed to persist reward to "
            f"{out_path}: {exc}\n"
        )


# Public API: realtime polymarket event prediction (deferred via ledger)


def fetch_active_markets(*, timeout_sec: float = _DEFAULT_TIMEOUT_SEC) -> list:
    """Return the trial's discovered Polymarket universe.

    Each entry carries the agent-facing market context (question text,
    description, current YES price, orderbook NBBO, end_date, tags,
    etc.) needed to make informed probability predictions. The list is
    frozen for the trial — re-calling returns the same set.

    See ``GET /markets/active`` in the polymarket curated handler for
    the exact response shape.
    """
    import requests

    url, token, _agent_id = _resolve_endpoint()
    resp = requests.get(
        url.rstrip("/") + "/markets/active",
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout_sec,
    )
    parsed = _parse_response(resp)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("markets"), list):
        return parsed["markets"]
    return []


def submit_event_predictions_async(
    predictions: list,
    *,
    submission_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    initial_retry_budget_sec: float = _INITIAL_RETRY_BUDGET_SEC,
) -> Dict[str, Any]:
    """Submit a batch of probability predictions for binary-event markets.

    Each prediction is a dict with::

        {"symbol": "0xabc…", "predicted_yes_probability": 0.7}

    The verifier captures the current YES price as ``entry_price``
    server-side, looks up the market's absolute resolution time from
    the discovered universe, and writes a pending row to the ledger.
    Returns synchronously with ``status="deferred"`` plus metadata
    identifying the session for later resolution.

    Resolution happens out-of-band via
    ``python -m openfinai_harbor.resolve_deferred <trial_dir>`` after
    every market in the session has resolved.
    """
    url, token, agent_id = _resolve_endpoint()
    sid = submission_id or str(uuid.uuid4())
    payload: Dict[str, Any] = {
        "agent_id": agent_id,
        "submission_id": sid,
        "predictions": list(predictions),
    }
    # Host trial dir back-pointer for orphan detection. Harbor sets
    # HARBOR_TRIAL_DIR on the agent container at trial spawn. Absent
    # outside harbor — the handler treats it as optional.
    trial_dir = os.environ.get("HARBOR_TRIAL_DIR")
    if trial_dir:
        payload["trial_dir"] = trial_dir
    if extra:
        payload["extra"] = dict(extra)
    return _do_post(
        url.rstrip("/") + "/submit/event_predictions_async",
        token,
        payload,
        timeout_sec=timeout_sec,
        initial_retry_budget_sec=initial_retry_budget_sec,
    )


def submit_event_predictions_async_and_persist(
    predictions: list,
    *,
    reward_output_path: Optional[str] = None,
    submission_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    initial_retry_budget_sec: float = _INITIAL_RETRY_BUDGET_SEC,
) -> Dict[str, Any]:
    """Submit event predictions and persist reward and resolver sidecars."""
    result = submit_event_predictions_async(
        predictions,
        submission_id=submission_id,
        extra=extra,
        timeout_sec=timeout_sec,
        initial_retry_budget_sec=initial_retry_budget_sec,
    )
    _persist_reward_json(result, reward_output_path)
    _persist_deferred_sidecar(result, reward_output_path)
    return result


# Heartbeat / deregister (shared across all curated families)


def heartbeat(*, timeout_sec: float = 5.0) -> Dict[str, Any]:
    url, token, agent_id = _resolve_endpoint()
    return _do_post(
        url.rstrip("/") + "/heartbeat",
        token,
        {"agent_id": agent_id},
        timeout_sec=timeout_sec,
    )


def deregister(*, timeout_sec: float = 5.0) -> Dict[str, Any]:
    try:
        url, token, agent_id = _resolve_endpoint()
    except RuntimeError:
        return {}
    try:
        return _do_post(
            url.rstrip("/") + "/deregister",
            token,
            {"agent_id": agent_id},
            timeout_sec=timeout_sec,
        )
    except Exception:
        return {}
