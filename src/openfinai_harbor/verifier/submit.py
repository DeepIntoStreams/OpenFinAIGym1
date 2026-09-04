"""Self-contained in-container client for the host verifier.

``submit`` reads the verifier URL, token, and agent ID from the environment and
sends encoded predictions using requests and a local NumPy import.
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

# NumPy remains a local import so host-side tooling need not install it.


# HARBOR_SUBMIT_TIMEOUT_SEC overrides the per-request timeout. Default 60s
# fits forecasting metrics but is tight for heavy generation scoring; per-call
# override via ``timeout_sec`` kwarg.
def _resolve_default_timeout_sec() -> float:
    raw = os.environ.get("HARBOR_SUBMIT_TIMEOUT_SEC")
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return 60.0


_DEFAULT_TIMEOUT_SEC = _resolve_default_timeout_sec()
_INITIAL_RETRY_BUDGET_SEC = 10.0
_INITIAL_RETRY_BACKOFF_SEC = 0.5


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


def _encode_array(arr: Any) -> str:
    """Encode an ndarray to base64 of its ``.npy`` bytes."""
    import numpy as np  # local import: see module docstring

    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resolve_endpoint() -> tuple[str, str, str]:
    """Read VERIFIER_URL + VERIFIER_TOKEN + AGENT_ID from the environment.

    AGENT_ID is auto-generated if absent (UUID4). Missing URL/TOKEN is
    fatal — agent code is supposed to be invoked with these set.
    """
    url = os.environ.get("VERIFIER_URL", "").strip()
    token = os.environ.get("VERIFIER_TOKEN", "").strip()
    if not url or not token:
        raise RuntimeError(
            "VERIFIER_URL and VERIFIER_TOKEN must be set in the agent's "
            "environment (set by `docker run -e ...` or compose env)."
        )
    agent_id = os.environ.get("AGENT_ID", "").strip() or str(uuid.uuid4())
    return url, token, agent_id


def _do_post(
    url: str,
    token: str,
    payload: Dict[str, Any],
    *,
    timeout_sec: float,
    initial_retry_budget_sec: float = 0.0,
) -> Dict[str, Any]:
    """POST JSON with optional initial-connection retry budget."""
    import requests

    deadline = time.time() + max(initial_retry_budget_sec, 0.0)
    backoff = _INITIAL_RETRY_BACKOFF_SEC
    last_exc: Optional[BaseException] = None
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
            last_exc = exc
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


def submit(
    predictions: Any,
    *,
    fake_emb: Optional[Any] = None,
    submission_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    initial_retry_budget_sec: float = _INITIAL_RETRY_BUDGET_SEC,
) -> Dict[str, Any]:
    """Submit ``predictions`` to the host verifier; return the parsed response.

    Returns a dict with keys ``scores``, ``weighted_total``,
    ``elapsed_sec``, ``submission_id``, ``cached``. Idempotent on
    ``submission_id`` — pass the same ID for retries to get the cached
    score back rather than re-running the metric.
    """
    url, token, agent_id = _resolve_endpoint()
    sid = submission_id or str(uuid.uuid4())
    payload: Dict[str, Any] = {
        "agent_id": agent_id,
        "submission_id": sid,
        "predictions_b64": _encode_array(predictions),
    }
    if fake_emb is not None:
        payload["fake_emb_b64"] = _encode_array(fake_emb)
    if extra:
        payload["extra"] = dict(extra)
    return _do_post(
        url.rstrip("/") + "/submit",
        token,
        payload,
        timeout_sec=timeout_sec,
        initial_retry_budget_sec=initial_retry_budget_sec,
    )


def submit_and_persist(
    predictions: Any,
    *,
    reward_output_path: Optional[str] = None,
    fake_emb: Optional[Any] = None,
    submission_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    initial_retry_budget_sec: float = _INITIAL_RETRY_BUDGET_SEC,
) -> Dict[str, Any]:
    """Convenience: submit + write reward.json next to the agent's runtime.

    The runner (``run_evaluation_explicit.py``) sets
    ``HARBOR_REWARD_OUTPUT`` in the agent's environment to control where
    the file lands; if neither that env nor ``reward_output_path`` is
    set, defaults to ``/logs/verifier/reward.json`` (Harbor's canonical
    verifier-output mount).
    """
    result = submit(
        predictions,
        fake_emb=fake_emb,
        submission_id=submission_id,
        extra=extra,
        timeout_sec=timeout_sec,
        initial_retry_budget_sec=initial_retry_budget_sec,
    )
    out_path = (
        reward_output_path
        or os.environ.get("HARBOR_REWARD_OUTPUT")
        or "/logs/verifier/reward.json"
    )
    # Two files: metrics.json (multi-key rich payload, per-channel scores)
    # + reward.json (strict single-key {"reward": float} for harbor's
    # Mean.compute which requires len(dict)==1). No reward.txt: harbor
    # reads reward_text_path first and the legacy single-float would mask
    # metrics.json updates.
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
    try:
        from pathlib import Path as _Path  # local import: see module docstring

        p = _Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        (p.parent / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )
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
            f"[harbor_submit] failed to persist reward to {out_path}: {exc}\n"
        )
    return result


def heartbeat(*, timeout_sec: float = 5.0) -> Dict[str, Any]:
    """POST /heartbeat for the current AGENT_ID. Idempotent."""
    url, token, agent_id = _resolve_endpoint()
    return _do_post(
        url.rstrip("/") + "/heartbeat",
        token,
        {"agent_id": agent_id},
        timeout_sec=timeout_sec,
    )


def deregister(*, timeout_sec: float = 5.0) -> Dict[str, Any]:
    """POST /deregister for the current AGENT_ID. Best-effort — silent on failure."""
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


def _cli() -> int:
    """Convenience CLI: submit a .npy file from the command line.

    Useful when a train.py wants a one-liner instead of importing this
    module. Reads the file, calls submit(), prints the JSON response.
    """
    import argparse

    p = argparse.ArgumentParser(prog="harbor_submit")
    p.add_argument("predictions_path", help=".npy / .npz / .csv file with predictions")
    p.add_argument("--fake-emb", default=None, help="Optional fake_emb .npy path")
    p.add_argument("--timeout-sec", type=float, default=_DEFAULT_TIMEOUT_SEC)
    args = p.parse_args()

    import numpy as np  # local import: keeps top-level cost down

    predictions = _load_array_for_cli(args.predictions_path)
    fake_emb = (
        _load_array_for_cli(args.fake_emb) if args.fake_emb else None
    )
    try:
        result = submit(
            predictions,
            fake_emb=fake_emb,
            timeout_sec=args.timeout_sec,
        )
    except Exception as exc:
        print(f"submit failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def _load_array_for_cli(path: str) -> Any:
    import numpy as np

    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        data = np.load(path)
        return data[list(data.files)[0]]
    if path.endswith(".csv"):
        return np.loadtxt(path, delimiter=",", skiprows=1, encoding="utf-8")
    raise ValueError(f"unrecognised file extension: {path}")


if __name__ == "__main__":
    sys.exit(_cli())
