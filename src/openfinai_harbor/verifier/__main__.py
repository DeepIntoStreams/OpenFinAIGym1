"""Entry point for ``python -m openfinai_harbor.verifier``.

Spawned by the per-task spawn coordinator with explicit ``--port``,
``--bind``, ``--token`` and ``--task-dir``. Detects whether the task is
auto-pipe (``manifest.json`` + ``evaluator.py``) or curated
(``task.toml`` with ``[metadata].family``) and builds the corresponding
FastAPI app. Stashes the uvicorn ``Server`` on ``app.state.server`` so
the lifecycle thread can request graceful shutdown, then runs.

This module is also runnable manually for debugging::

    python -m openfinai_harbor.verifier \
        --task-dir tasks/generated/alpha_trading/<task> \
        --bind 127.0.0.1 --port 5000 \
        --token <hex> --despawn-grace-min 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import uvicorn

from openfinai_harbor.verifier import registry
from openfinai_harbor.verifier.curated import (
    CuratedVerifierConfig,
    create_curated_app,
)
from openfinai_harbor.verifier.curated.task_loader import (
    is_curated_bundle,
    load_curated_spec,
)
from openfinai_harbor.verifier.server import VerifierConfig, create_app


_DEFAULT_BIND = "127.0.0.1"
_DEFAULT_PORT = 0  # 0 means OS-assigned (fallback for direct invocation)
_DEFAULT_DESPAWN_GRACE_MIN = 10.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m openfinai_harbor.verifier",
        description="Per-task verifier RPC service for OpenFinGym.",
    )
    p.add_argument(
        "--task-dir",
        required=True,
        type=Path,
        help="Path to the installed task bundle (tasks/generated/<scope>/<task>/).",
    )
    p.add_argument("--bind", default=_DEFAULT_BIND)
    p.add_argument("--port", type=int, default=_DEFAULT_PORT)
    p.add_argument(
        "--token",
        required=True,
        help="Bearer token clients must include as Authorization header.",
    )
    p.add_argument(
        "--despawn-grace-min",
        type=float,
        default=_DEFAULT_DESPAWN_GRACE_MIN,
        help=(
            "Minutes the verifier keeps running after the last agent "
            "deregisters / heartbeat lapses. 0 = immediate shutdown."
        ),
    )
    p.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help=(
            "Override per-task state dir. Defaults to "
            "data/run_output/verifier/<task_id>/."
        ),
    )
    p.add_argument(
        "--heartbeat-timeout-sec", type=float, default=90.0,
    )
    p.add_argument(
        "--liveness-poll-sec", type=float, default=30.0,
    )
    p.add_argument(
        "--max-requests-per-minute", type=int, default=60,
    )
    p.add_argument(
        "--submission-cache-size", type=int, default=64,
    )
    p.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    return p.parse_args()


def _resolve_task_metadata(task_dir: Path) -> tuple[str, bool]:
    """Read manifest.json (or fall back to defaults) for (interaction_model, has_held_out_test_gt).

    Manifest is the canonical descriptor written by Phase 4 and
    re-patched by ``install_task`` from ``slice_result.shape`` once the
    loader has run. If absent (e.g. legacy task), default to
    forecasting + True so the verifier at least starts up.
    """
    candidates = [
        task_dir / "environment" / "data" / "manifest.json",
        task_dir / "manifest.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            return (
                str(payload.get("interaction_model") or "forecasting"),
                bool(payload.get("has_held_out_test_gt", True)),
            )
    return "forecasting", True


def _build_config(args: argparse.Namespace) -> VerifierConfig:
    task_dir: Path = args.task_dir.resolve()
    if not task_dir.exists():
        raise SystemExit(f"task_dir does not exist: {task_dir}")
    data_dir = task_dir / "environment" / "data"
    eval_data_dir = task_dir / "environment" / "eval-data"
    evaluator_path = data_dir / "evaluator.py"
    if not evaluator_path.exists():
        raise SystemExit(f"evaluator not found: {evaluator_path}")
    task_id = task_dir.name
    state_dir = (
        args.state_dir.resolve()
        if args.state_dir is not None
        else registry.state_dir_for_task(task_id=task_id)
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    run_dir = registry.new_run_dir(state_dir)

    interaction_model, has_held_out_test_gt = _resolve_task_metadata(task_dir)

    return VerifierConfig(
        task_id=task_id,
        task_dir=task_dir,
        data_dir=data_dir,
        eval_data_dir=eval_data_dir,
        evaluator_path=evaluator_path,
        interaction_model=interaction_model,
        has_held_out_test_gt=has_held_out_test_gt,
        state_dir=state_dir,
        run_dir=run_dir,
        token=args.token,
        despawn_grace_min=args.despawn_grace_min,
        heartbeat_timeout_sec=args.heartbeat_timeout_sec,
        liveness_poll_sec=args.liveness_poll_sec,
        max_requests_per_minute=args.max_requests_per_minute,
        submission_cache_size=args.submission_cache_size,
    )


def _build_curated_config(args: argparse.Namespace) -> CuratedVerifierConfig:
    """Build a curated VerifierConfig from CLI flags + ``task.toml``.

    Mirrors :func:`_build_config` but reads dispatch info from
    ``task.toml`` instead of ``manifest.json`` / ``evaluator.py``.
    """
    task_dir: Path = args.task_dir.resolve()
    if not task_dir.exists():
        raise SystemExit(f"task_dir does not exist: {task_dir}")
    spec = load_curated_spec(task_dir)

    state_dir = (
        args.state_dir.resolve()
        if args.state_dir is not None
        else registry.state_dir_for_task(task_id=spec.task_id)
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    run_dir = registry.new_run_dir(state_dir)

    return CuratedVerifierConfig(
        spec=spec,
        state_dir=state_dir,
        run_dir=run_dir,
        token=args.token,
        despawn_grace_min=args.despawn_grace_min,
        heartbeat_timeout_sec=args.heartbeat_timeout_sec,
        liveness_poll_sec=args.liveness_poll_sec,
        max_requests_per_minute=args.max_requests_per_minute,
        submission_cache_size=args.submission_cache_size,
    )


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Dispatch: curated if task.toml declares a recognized family, else
    # auto-pipe. We resolve task_dir defensively here (rather than
    # delegating to the per-flavor _build_config) so the dispatch
    # decision is logged before the heavy startup work begins — easier
    # to diagnose "why did the wrong app type spawn".
    task_dir_resolved = args.task_dir.resolve()
    is_curated = task_dir_resolved.exists() and is_curated_bundle(
        task_dir_resolved
    )

    config: Any
    app: Any
    try:
        if is_curated:
            curated_cfg = _build_curated_config(args)
            logging.info(
                "dispatching CURATED verifier: task_id=%s family=%s",
                curated_cfg.task_id,
                curated_cfg.family,
            )
            config = curated_cfg
            app = create_curated_app(curated_cfg)
        else:
            auto_cfg = _build_config(args)
            logging.info(
                "dispatching AUTO-PIPE verifier: task_id=%s",
                auto_cfg.task_id,
            )
            config = auto_cfg
            app = create_app(auto_cfg)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        return 2
    uv_config = uvicorn.Config(
        app=app,
        host=args.bind,
        port=args.port,
        log_level=args.log_level,
        access_log=True,
        loop="asyncio",
        # Don't install signal handlers for SIGINT in test runners.
        # In normal runs uvicorn picks them up by default; we keep that.
    )
    server = uvicorn.Server(uv_config)
    # Stash the server reference BEFORE serving so the lifecycle thread
    # (started in lifespan) can flip should_exit for graceful shutdown.
    app.state.server = server

    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    except Exception:
        # Last-ditch: write crash marker before bubbling up so operators
        # have a single artefact to inspect.
        try:
            crash_path = registry.crashed_marker_path(config.run_dir)
            crash_path.write_text(
                json.dumps(
                    {
                        "task_id": config.task_id,
                        "ts_utc": registry.utc_now_iso(),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
        traceback.print_exc()
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
