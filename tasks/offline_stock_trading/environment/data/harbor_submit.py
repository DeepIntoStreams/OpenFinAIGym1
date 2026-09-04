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


# Interactive offline trading
# The verifier exposes one observation per committed action, keeping future
# bars and scoring server-side. Realtime bundles use their in-container loop.


def session_reset(
    *,
    session_id: Optional[str] = None,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    initial_retry_budget_sec: float = _INITIAL_RETRY_BUDGET_SEC,
) -> Dict[str, Any]:
    """Open an interactive trading episode; return the first observation.

    Response keys: ``session_id`` (echoed/minted), ``observation`` (a causal
    market + portfolio dict), ``step`` (0), ``done`` (False).
    """
    url, token, agent_id = _resolve_endpoint()
    payload: Dict[str, Any] = {"agent_id": agent_id}
    if session_id:
        payload["session_id"] = session_id
    return _do_post(
        url.rstrip("/") + "/session/reset",
        token,
        payload,
        timeout_sec=timeout_sec,
        initial_retry_budget_sec=initial_retry_budget_sec,
    )


def session_step(
    session_id: str,
    action: Any,
    *,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """Submit one action and advance one bar.

    Returns the next ``observation`` (``None`` on the terminal step), the
    realized ``reward``, ``done``, ``step``, and — on the terminal step —
    the server-computed ``scores`` + ``weighted_total``.
    """
    url, token, agent_id = _resolve_endpoint()
    payload = {"agent_id": agent_id, "session_id": session_id, "action": action}
    return _do_post(
        url.rstrip("/") + "/session/step",
        token,
        payload,
        timeout_sec=timeout_sec,
    )


def run_trading_session(
    policy: Any,
    *,
    max_steps: int = 1_000_000,
    reward_output_path: Optional[str] = None,
    session_id: Optional[str] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """Run a full interactive episode and (optionally) write reward.json.

    ``policy`` is a callable ``policy(observation) -> action`` invoked once
    per bar with the *current* observation (closes only up to now — no
    look-ahead is possible). The loop ends when the verifier reports
    ``done``; the server-side ``scores`` are then persisted. Returns the
    terminal step response (carrying ``scores`` / ``weighted_total``).
    """
    reset = session_reset(session_id=session_id)
    sid = reset["session_id"]
    obs = reset.get("observation")
    done = bool(reset.get("done", False))
    final: Dict[str, Any] = reset
    steps = 0
    while not done and steps < max_steps:
        action = policy(obs)
        final = session_step(sid, action)
        obs = final.get("observation")
        done = bool(final.get("done", False))
        steps += 1
    if persist:
        _persist_reward_json(
            {
                "scores": final.get("scores") or {},
                "weighted_total": final.get("weighted_total", 0.0),
                "status": "scored",
            },
            reward_output_path,
        )
    return final


# Deferred realtime forecasting


def submit_predictions_async(
    predictions: list,
    *,
    submission_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    initial_retry_budget_sec: float = _INITIAL_RETRY_BUDGET_SEC,
) -> Dict[str, Any]:
    """Submit deferred price/direction predictions.

    Each item requires a symbol and either a predicted price or direction. The
    verifier captures entry prices and returns ledger metadata for resolution.
    """
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
    """Write canonical reward.json and the numeric metrics.json sidecar.

    ``weighted_total`` becomes the single reward value; score channels and
    status markers remain in metrics.json. reward.txt is omitted because Harbor
    gives it precedence over reward.json.
    """
    out_path = (
        reward_output_path
        or os.environ.get("HARBOR_REWARD_OUTPUT")
        or "/logs/verifier/reward.json"
    )

    metrics_payload: Dict[str, float] = {}
    scores = result.get("scores") or {}
    if isinstance(scores, dict):
        for k, v in scores.items():
            if not isinstance(v, bool) and isinstance(v, (int, float)):
                metrics_payload[k] = v
    weighted_total = result.get("weighted_total", 0.0)
    if not isinstance(weighted_total, bool) and isinstance(
        weighted_total, (int, float)
    ):
        metrics_payload["reward"] = weighted_total
    else:
        metrics_payload["reward"] = 0.0

    sentinel_status = result.get("status")
    if sentinel_status == "scored":
        metrics_payload["status_scored"] = 1.0
    elif sentinel_status == "deferred":
        metrics_payload["status_deferred"] = 1.0

    try:
        from pathlib import Path as _Path

        p = _Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # metrics.json — rich multi-key payload.
        (p.parent / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )
        # reward.json — strict single-key for harbor's Mean aggregator.
        reward_scalar = metrics_payload.get("reward", 0.0)
        try:
            reward_scalar = float(reward_scalar)
        except (TypeError, ValueError):
            reward_scalar = 0.0
        p.write_text(
            json.dumps({"reward": reward_scalar}, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        sys.stderr.write(
            f"[harbor_submit] failed to persist reward to "
            f"{out_path}: {exc}\n"
        )


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
