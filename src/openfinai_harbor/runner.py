"""Plan and configure Harbor trials for curated task bundles.

Helpers centralize verifier selection, provider credentials, container
environment, mounts, and score display across offline and realtime families.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover — Python 3.10 fallback
    import tomli as tomllib  # type: ignore[import-not-found]


__all__ = [
    "CuratedTaskPlan",
    "inspect_curated_task",
    "preflight_curated_credentials",
    "build_container_env",
    "build_task_data_mount",
    "pretty_scores",
    "maybe_emit_market_closed_warning",
]


@dataclass(frozen=True)
class CuratedTaskPlan:
    """Snapshot of the per-task branching decisions a host orchestrator needs.

    Built once from ``task.toml`` by :func:`inspect_curated_task` and then
    passed to :func:`preflight_curated_credentials` and
    :func:`build_container_env`. Keeping it as a dataclass means a future
    sixth family adds one field here and one branch in each consumer,
    rather than scattering ``family == "..."`` checks across callers.
    """

    family: Optional[str]
    class_name: Optional[str]
    skip_rpc_verifier: bool
    needs_alpaca_creds: bool
    submit_timeout_sec: Optional[float]


def _coerce_positive_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _read_task_meta(task_dir: Path) -> dict[str, Any]:
    """Read task-level orchestration fields from ``task.toml``.

    Returns a dict with keys ``family`` (str | None) and ``class_name``
    (str | None). A missing/unparseable file yields the all-``None``
    shape — callers should treat that as "no task-specific orchestration
    overrides apply" rather than an error.
    """
    toml_path = task_dir / "task.toml"
    if not toml_path.exists():
        return {"family": None, "class_name": None, "submit_timeout_sec": None}
    try:
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except Exception:
        return {"family": None, "class_name": None, "submit_timeout_sec": None}
    metadata = raw.get("metadata") or {}
    curated = raw.get("curated") or {}
    environment = raw.get("environment") or {}
    return {
        "family": metadata.get("family"),
        "class_name": curated.get("class_name"),
        "submit_timeout_sec": _coerce_positive_float(
            environment.get("submit_timeout_sec")
        ),
    }


def _needs_alpaca_creds(class_name: Optional[str]) -> bool:
    """Stock realtime bundles import AlpacaProvider in-container.

    They need ALPACA_API_KEY / ALPACA_SECRET_KEY plumbed in via the
    verifier env (which docker run forwards). Crypto bundles use
    Binance public API and need no creds.
    """
    if not class_name:
        return False
    return class_name.startswith("RealtimeStock")


def _is_realtime_trading_bundle(family: Optional[str]) -> bool:
    """Realtime trading bundles run the gym loop entirely in the agent
    container; the host RPC verifier is not spawned (there's nothing
    for the agent to submit to). Note: harbor's verifier STEP still
    runs (executing tests/test.sh) — that's how the bundled
    run_evaluation_curated.py + train.py get invoked. We only skip the
    separate RPC verifier process; verifier.disable stays False.
    """
    return family == "realtime_trading"


def inspect_curated_task(task_dir: Path) -> CuratedTaskPlan:
    """Read ``task.toml`` and derive the per-task orchestration plan."""
    meta = _read_task_meta(task_dir)
    family = meta.get("family")
    class_name = meta.get("class_name")
    return CuratedTaskPlan(
        family=family,
        class_name=class_name,
        skip_rpc_verifier=_is_realtime_trading_bundle(family),
        needs_alpaca_creds=_needs_alpaca_creds(class_name),
        submit_timeout_sec=meta.get("submit_timeout_sec"),
    )


def preflight_curated_credentials(plan: CuratedTaskPlan) -> Optional[str]:
    """Return an error message if required credentials are missing, else None.

    Called before spawning verifier / building trial config so the
    operator gets a friendly abort instead of burning compute on a
    doomed trial. Only stock realtime bundles need creds today.
    """
    if not plan.needs_alpaca_creds:
        return None
    if not os.environ.get("ALPACA_API_KEY") or not os.environ.get(
        "ALPACA_SECRET_KEY"
    ):
        return (
            f"task class {plan.class_name!r} requires Alpaca paper-trading "
            "credentials in your environment (the agent container fetches "
            "live SPY bars and dispatches paper orders). Set "
            "ALPACA_API_KEY and ALPACA_SECRET_KEY in your shell or .env "
            "file. Free tier suffices: https://alpaca.markets/algotrading"
        )
    return None


def build_container_env(
    *,
    plan: CuratedTaskPlan,
    container_url: str,
    token: str,
    agent_id: str,
    trial_dir: Path | None = None,
) -> dict[str, str]:
    """Compose the env dict forwarded into the agent container.

    Always includes AGENT_ID. VERIFIER_URL/VERIFIER_TOKEN are skipped
    when the RPC verifier wasn't spawned — the in-container runner has
    no host endpoint to call. ALPACA_API_KEY/SECRET are forwarded only
    when the bundle actually needs them; preflight will have already
    aborted if they are missing. HARBOR_TRIAL_DIR (host absolute path)
    lets the in-container submit helper attach a trial back-pointer to
    each ledger row so post-trial inspection / cleanup tools can detect
    orphaned ledgers without scanning the examples/ tree.
    """
    env: dict[str, str] = {"AGENT_ID": agent_id}
    if not plan.skip_rpc_verifier:
        env["VERIFIER_URL"] = container_url
        env["VERIFIER_TOKEN"] = token
    if plan.needs_alpaca_creds:
        env["ALPACA_API_KEY"] = os.environ.get("ALPACA_API_KEY", "")
        env["ALPACA_SECRET_KEY"] = os.environ.get("ALPACA_SECRET_KEY", "")
    if trial_dir is not None:
        env["HARBOR_TRIAL_DIR"] = str(trial_dir.resolve())
    # Forward HARBOR_SUBMIT_TIMEOUT_SEC into the container so heavy
    # generation metrics (sig-MMD on thousands of samples) can exceed
    # the 60s default.
    submit_timeout = plan.submit_timeout_sec
    if submit_timeout is not None:
        env["HARBOR_SUBMIT_TIMEOUT_SEC"] = str(submit_timeout)
    else:
        submit_timeout = os.environ.get("HARBOR_SUBMIT_TIMEOUT_SEC")
        if submit_timeout:
            env["HARBOR_SUBMIT_TIMEOUT_SEC"] = submit_timeout
    return env


def build_task_data_mount(task_dir: Path) -> Optional[dict[str, str]]:
    """Return the Harbor `mounts` entry that exposes a task's data at /data.

    The shared sandbox image (``nihao0630/openfinai-base:v1``) carries
    no task-specific payload — Harbor's mount mechanism injects
    ``<task>/environment/data`` into the agent sandbox at trial-launch.
    Both Docker and Singularity providers honor the same dict shape
    (Harbor TrialConfig.environment.mounts: see harbor.trial.trial.
    Trial._default_agent_env_mounts, which appends user-supplied mounts
    on top of Harbor's own /logs/* binds).

    Returns ``None`` when the task has no ``environment/data`` directory,
    so callers can ``filter(None, [...])`` the result into the mounts
    list without conditionals. Auto-pipeline tasks without bundled data
    fall into this case before the loader stage runs.
    """
    data_dir = (task_dir / "environment" / "data").resolve()
    if not data_dir.is_dir():
        return None
    return {
        "type": "bind",
        "source": data_dir.as_posix(),
        "target": "/data",
    }


def pretty_scores(scores: Optional[dict[str, Any]]) -> str:
    """Format a ``rewards`` dict for terminal output.

    Skips private keys (leading underscore), sorts the rest, and renders
    floats with six decimal places. Returns ``"(none)"`` when the dict
    is empty or contains only private keys.
    """
    if not scores:
        return "(none)"
    parts: list[str] = []
    for k in sorted(scores.keys()):
        if isinstance(k, str) and k.startswith("_"):
            continue
        v = scores[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.6f}")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts) or "(none)"


def maybe_emit_market_closed_warning(
    rewards: Optional[dict[str, Any]],
    trial_dir: Path,
    *,
    log_prefix: str = "[openfinai]",
) -> None:
    """Promote the pipeline's ``market_closed=1.0`` signal to a terminal banner.

    The pipeline (``agent_runtime.py``) already writes a structured
    sentinel in ``reward.json`` plus a human-readable ``error.txt`` plus
    a one-line stderr warning from inside the container. This helper
    surfaces that condition prominently on the host so a developer
    running an example or sweep at a terminal sees it without having to
    open the trial folder. Tooling that consumes ``reward.json``
    directly is unaffected — they can keep gating on the structured
    ``market_closed`` field.
    """
    import sys

    if not rewards or float(rewards.get("market_closed", 0.0)) < 1.0:
        return
    staleness = rewards.get("max_observed_staleness_seconds")
    staleness_str = (
        f"latest bar ~{float(staleness) / 3600:.1f}h stale"
        if isinstance(staleness, (int, float))
        else "feed stale"
    )
    error_txt = trial_dir / "verifier" / "error.txt"
    print(
        f"{log_prefix} WARNING: market appears closed — "
        f"{staleness_str}. Trial scored 0; this is a clean abort, "
        "not an agent failure.",
        file=sys.stderr,
        flush=True,
    )
    if error_txt.is_file():
        print(
            f"{log_prefix}          see {error_txt} for the full diagnostic.",
            file=sys.stderr,
            flush=True,
        )
