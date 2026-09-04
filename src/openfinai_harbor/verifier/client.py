"""Host-side Python client for the verifier RPC service.

Used by ``openfinai_harbor.run_trial`` and any future host-side
orchestrator (the in-container helper is the separate ``submit.py``
file with a deliberately smaller dependency surface). This module is
free to import everything from ``openfinai_harbor.verifier`` since the
host-side caller already has the package installed.

Public surface::

    endpoint = client.get_or_spawn(task_dir, despawn_grace_min=10)
    info = client.info(endpoint)                      # GET /info
    client.register(endpoint, agent_id)               # POST /register
    response = client.submit(endpoint, agent_id, predictions)  # POST /submit
    client.deregister(endpoint, agent_id)             # POST /deregister
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import requests

from openfinai_harbor.verifier.models import (
    HeartbeatResponse,
    InfoResponse,
    RegisterResponse,
    SubmitResponse,
)
from openfinai_harbor.verifier.spawn import (
    VerifierEndpoint,
    VerifierSpawnError,
    get_or_spawn as _spawn_get_or_spawn,
    request_despawn,
)


__all__ = [
    "VerifierEndpoint",
    "VerifierSpawnError",
    "VerifierUnreachableError",
    "VerifierAuthError",
    "VerifierRateLimitedError",
    "VerifierScoringError",
    "container_url_for",
    "encode_array",
    "get_or_spawn",
    "info",
    "register",
    "heartbeat",
    "deregister",
    "resolve_url_mode",
    "submit",
    "request_despawn",
]


def resolve_url_mode() -> str:
    """Decide how the agent container reaches the host verifier loopback.

    Honors ``OPENFINAI_VERIFIER_URL_MODE`` when set to ``rewrite`` or
    ``host``. Otherwise picks an OS-aware default:

    * Docker Desktop hosts (``win32`` / ``darwin``) → ``rewrite``.
      Containers run inside a Linux VM, so ``network_mode: host`` would
      share the *VM's* loopback (not the host's); Docker Desktop ships
      a ``host.docker.internal`` DNS shim that IS the host loopback,
      which is what we use.
    * Linux (``linux``, etc.) → ``host``. Covers rootless podman and
      plain Linux docker, where ``host.docker.internal`` is not defined
      and the bridge/slirp4netns gateway isn't directly routable; the
      compose override pins ``network_mode: host`` so ``127.0.0.1``
      inside the container IS the host loopback.
    """
    raw = os.environ.get("OPENFINAI_VERIFIER_URL_MODE", "").strip().lower()
    if raw in ("rewrite", "host"):
        return raw
    return "rewrite" if sys.platform in ("win32", "darwin") else "host"


def container_url_for(host_url: str) -> str:
    """Translate a host loopback URL to one the agent container can reach.

    Two strategies, picked by :func:`resolve_url_mode`:

    * ``rewrite`` — rewrite ``127.0.0.1`` / ``0.0.0.0`` to
      ``host.docker.internal``. Used on Docker Desktop (Win/Mac) by
      default; Docker Desktop's DNS shim makes that name resolve to the
      host loopback. Also the right pick when the operator wants
      bridge networking on a hand-rolled compose override.
    * ``host`` — return the URL unchanged. Used on Linux by default and
      requires the matching ``environment/docker-compose.yaml`` override
      (``network_mode: host``) written by
      ``openfinai_harbor.run_trial._ensure_host_network_compose``.

    Override at run time: ``OPENFINAI_VERIFIER_URL_MODE=rewrite|host``.
    """
    if resolve_url_mode() == "rewrite":
        return host_url.replace("127.0.0.1", "host.docker.internal").replace(
            "0.0.0.0", "host.docker.internal"
        )
    return host_url


class VerifierUnreachableError(RuntimeError):
    """Raised when the verifier endpoint is not reachable."""


class VerifierAuthError(RuntimeError):
    """401 from the verifier — token is missing or wrong."""


class VerifierRateLimitedError(RuntimeError):
    """429 from the verifier — too many submissions in the window."""

    def __init__(self, message: str, *, retry_after_sec: float) -> None:
        super().__init__(message)
        self.retry_after_sec = retry_after_sec


class VerifierScoringError(RuntimeError):
    """500 from the verifier — evaluator.score() raised."""


def get_or_spawn(
    task_dir: Path,
    *,
    despawn_grace_min: float = 10.0,
    bind: str = "127.0.0.1",
    run_output_root: Optional[Path] = None,
    max_requests_per_minute: int | None = None,
) -> VerifierEndpoint:
    """Spawn a verifier for ``task_dir`` if none is running, else attach.

    ``max_requests_per_minute`` overrides the verifier's per-token rate
    limit (default 60/min); pass a large value when prewarming for an
    RL training run that will submit at high frequency.
    """
    return _spawn_get_or_spawn(
        task_dir,
        despawn_grace_min=despawn_grace_min,
        bind=bind,
        run_output_root=run_output_root,
        max_requests_per_minute=max_requests_per_minute,
    )


def encode_array(arr: np.ndarray) -> str:
    """Encode a numpy array as base64 of its ``np.save`` bytes (round-trips shape+dtype)."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _post(
    endpoint: VerifierEndpoint,
    path: str,
    *,
    payload: Mapping[str, Any],
    timeout_sec: float,
    initial_retry_budget_sec: float = 0.0,
) -> dict[str, Any]:
    """POST JSON to ``path``; surface verifier errors as typed exceptions."""
    url = endpoint.url.rstrip("/") + path
    deadline = time.time() + max(initial_retry_budget_sec, 0.0)
    backoff = 0.5
    last_exc: Optional[Exception] = None
    while True:
        try:
            resp = requests.post(
                url,
                json=dict(payload),
                headers={
                    **endpoint.auth_header(),
                    "Content-Type": "application/json",
                },
                timeout=timeout_sec,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if time.time() >= deadline:
                raise VerifierUnreachableError(
                    f"verifier {endpoint.task_id} unreachable at {url}: {exc}"
                ) from exc
            time.sleep(min(backoff, max(0.1, deadline - time.time())))
            backoff = min(backoff * 1.5, 2.0)
            continue
        return _parse_response(resp, endpoint=endpoint)


def _get(
    endpoint: VerifierEndpoint,
    path: str,
    *,
    timeout_sec: float,
    auth_required: bool = True,
) -> dict[str, Any]:
    url = endpoint.url.rstrip("/") + path
    headers = endpoint.auth_header() if auth_required else {}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_sec)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise VerifierUnreachableError(
            f"verifier {endpoint.task_id} unreachable at {url}: {exc}"
        ) from exc
    return _parse_response(resp, endpoint=endpoint)


def _parse_response(resp: requests.Response, *, endpoint: VerifierEndpoint) -> dict[str, Any]:
    if resp.status_code == 401:
        raise VerifierAuthError(
            f"verifier {endpoint.task_id} rejected token (401)"
        )
    if resp.status_code == 429:
        retry_after_str = resp.headers.get("Retry-After", "0")
        try:
            retry_after = float(retry_after_str)
        except ValueError:
            retry_after = 0.0
        raise VerifierRateLimitedError(
            f"verifier rate-limited (429); retry-after={retry_after}s",
            retry_after_sec=retry_after,
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


def info(
    endpoint: VerifierEndpoint, *, timeout_sec: float = 5.0
) -> InfoResponse:
    return InfoResponse(**_get(endpoint, "/info", timeout_sec=timeout_sec))


def register(
    endpoint: VerifierEndpoint,
    agent_id: Optional[str] = None,
    *,
    notes: Optional[str] = None,
    timeout_sec: float = 5.0,
    initial_retry_budget_sec: float = 10.0,
) -> tuple[str, RegisterResponse]:
    """Register an agent. Returns ``(agent_id, response)``.

    If ``agent_id`` is None, generates a UUID4. The first call after
    spawn pays the initial-retry budget so the lazy-spawn ready window
    is absorbed transparently.
    """
    aid = agent_id or str(uuid.uuid4())
    payload: dict[str, Any] = {"agent_id": aid}
    if notes is not None:
        payload["notes"] = notes
    raw = _post(
        endpoint,
        "/register",
        payload=payload,
        timeout_sec=timeout_sec,
        initial_retry_budget_sec=initial_retry_budget_sec,
    )
    return aid, RegisterResponse(**raw)


def heartbeat(
    endpoint: VerifierEndpoint, agent_id: str, *, timeout_sec: float = 5.0
) -> HeartbeatResponse:
    raw = _post(
        endpoint,
        "/heartbeat",
        payload={"agent_id": agent_id},
        timeout_sec=timeout_sec,
    )
    return HeartbeatResponse(**raw)


def deregister(
    endpoint: VerifierEndpoint, agent_id: str, *, timeout_sec: float = 5.0
) -> HeartbeatResponse:
    raw = _post(
        endpoint,
        "/deregister",
        payload={"agent_id": agent_id},
        timeout_sec=timeout_sec,
    )
    return HeartbeatResponse(**raw)


def submit(
    endpoint: VerifierEndpoint,
    agent_id: str,
    predictions: np.ndarray,
    *,
    fake_emb: Optional[np.ndarray] = None,
    submission_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    timeout_sec: float = 60.0,
) -> SubmitResponse:
    """POST /submit. ``predictions`` (and optionally ``fake_emb``) are encoded inline."""
    sid = submission_id or str(uuid.uuid4())
    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "submission_id": sid,
        "predictions_b64": encode_array(predictions),
    }
    if fake_emb is not None:
        payload["fake_emb_b64"] = encode_array(fake_emb)
    if extra:
        payload["extra"] = dict(extra)
    raw = _post(
        endpoint,
        "/submit",
        payload=payload,
        timeout_sec=timeout_sec,
    )
    return SubmitResponse(**raw)
