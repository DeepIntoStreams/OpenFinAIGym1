"""Run a Harbor trial with task data and, when needed, a host verifier.

The verifier keeps held-out targets on the host and exposes authenticated
scoring to the agent container. Realtime trading bundles score in-container
and skip that service.

Usage::

    python -m openfinai_harbor.run_trial \\
        --task-dir tasks/offline_stock_forecasting \\
        --agent-import-path openfinai_harbor.agents:SingleShotLLMAgent \\
        --provider ollama \\
        --model qwen2.5-coder:14b \\
        --trials-dir jobs/host_rpc_smoke

"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


# Harbor's Job creates Progress+Live without an explicit console (Rich owns
# wrapped stdout); we mirror that to avoid two-console rendering fights.
# _CONSOLE is only used post-Live for the summary table.
_CONSOLE = Console()


def _load_dotenv_best_effort() -> None:
    """Load API keys from .env at the repo root.

    PowerShell does not auto-load .env the way bash with direnv does, so
    invoking ``python -m openfinai_harbor.run_trial`` from a fresh shell
    would otherwise miss ``OPENROUTER_API_KEY`` and friends and fail
    deep inside LiteLLM.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    repo_root = Path(__file__).resolve().parents[2]
    repo_env = repo_root / ".env"
    if repo_env.exists():
        load_dotenv(dotenv_path=repo_env, override=False)
    load_dotenv(override=False)


_load_dotenv_best_effort()


def _utc_trial_name() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _format_provider_model_id(provider: str, model: str) -> str:
    """Build the LiteLLM model-id string from (provider, bare_model_name).

    LiteLLM identifies models as ``<provider>/<model>``; the same prefix
    drives BaseFinAgent's API-key fail-fast check. We accept either the
    bare model name + a ``--provider`` flag, or a fully-qualified
    ``provider/model`` string in ``--model`` (in which case the prefix
    is kept verbatim).
    """
    p = (provider or "").strip().lower()
    m = (model or "").strip()
    if not p:
        return m
    if m.startswith(f"{p}/"):
        return m
    return f"{p}/{m}"


def _parse_kv(items: list[str]) -> Dict[str, Any]:
    """Parse repeated ``--ak key=value`` flags into a kwargs dict.

    Why this exists: ``run_trial.py`` is agent-class-agnostic — the
    caller picks an agent via ``--agent-import-path`` and that agent's
    ``__init__`` decides which kwargs are valid. We cannot enumerate
    them at the CLI level, so we mirror harbor's own CLI convention
    (``harbor run … --ak max_tokens=8000 --ak temperature=0.0``) and let
    the agent's signature do the validation.

    Values are auto-coerced: a JSON object/array (``{...}`` / ``[...]``)
    parses to a dict/list — so nested kwargs like
    ``model_info={"max_input_tokens":8192,...}`` reach the agent intact —
    then ``"true"`` / ``"false"`` → bool, then int, then float, otherwise
    kept as str. Coercion is type-only — a path like
    ``examples/prompts/financial_simple.txt`` stays a string because none
    of the JSON/numeric/bool casts succeed.
    """
    out: Dict[str, Any] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"--ak must be key=value (got {raw!r})")
        k, v = raw.split("=", 1)
        k = k.strip()
        v = v.strip()
        # JSON object/array → dict/list (nested agent kwargs). Restricted to
        # the `{`/`[` lead so scalar strings still flow through the casts below.
        if v[:1] in "{[":
            try:
                out[k] = json.loads(v)
                continue
            except (ValueError, TypeError):
                pass
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def _render_trial_summary(
    *,
    result: Any,
    trial_name: str,
    trial_dir: Path,
    rewards: Optional[dict[str, Any]],
) -> None:
    """Print a Rich summary table for the trial (replaces print-based output).

    Shape mirrors harbor's ``print_job_results_tables`` for visual parity
    — a title line (``trial • agent • model``), a metrics table with
    one row per reward key + tokens/cost, and an exception table when
    the trial errored. On a non-TTY the underlying console renders
    plain text.
    """
    agent_name: Optional[str] = None
    model_name: Optional[str] = None
    if getattr(result, "agent_info", None) is not None:
        agent_name = getattr(result.agent_info, "name", None)
        if getattr(result.agent_info, "model_info", None) is not None:
            model_name = getattr(result.agent_info.model_info, "name", None)
    title_bits = [f"trial {trial_name}"]
    if agent_name:
        title_bits.append(agent_name)
    if model_name:
        title_bits.append(model_name)
    _CONSOLE.print()
    _CONSOLE.print(f"[bold]{' • '.join(title_bits)}[/bold]")

    metrics_table = Table(show_header=True)
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value", justify="right")
    if rewards:
        for key in sorted(rewards.keys()):
            if isinstance(key, str) and key.startswith("_"):
                continue  # internal fields (_error, _errors) — not metrics
            value = rewards[key]
            if isinstance(value, float):
                metrics_table.add_row(key, f"{value:.6f}")
            else:
                metrics_table.add_row(key, repr(value))
    else:
        metrics_table.add_row("[dim](no rewards)[/dim]", "")

    agent_result = getattr(result, "agent_result", None)
    if agent_result is not None:
        tokens = (
            f"{getattr(agent_result, 'n_input_tokens', 0)}/"
            f"{getattr(agent_result, 'n_output_tokens', 0)}"
        )
        metrics_table.add_row("tokens (in/out)", tokens)
        cost = getattr(agent_result, "cost_usd", None)
        if cost is not None:
            metrics_table.add_row("cost_usd", f"{float(cost):.6f}")
    _CONSOLE.print(metrics_table)

    exc = getattr(result, "exception_info", None)
    if exc is not None:
        exc_table = Table(show_header=True)
        exc_table.add_column("Exception")
        exc_table.add_column("Message")
        exc_table.add_row(
            getattr(exc, "exception_type", "?") or "?",
            (getattr(exc, "exception_message", "") or "").splitlines()[0][:200],
        )
        _CONSOLE.print()
        _CONSOLE.print(exc_table)

    _CONSOLE.print()
    _CONSOLE.print(f"[dim]result.json -> {trial_dir / 'result.json'}[/dim]")


_AUTO_COMPOSE_MARKER = "Auto-generated by openfinai_harbor.run_trial"


def _ensure_host_network_compose(task_dir: Path) -> None:
    """Manage the docker-compose override that pins ``network_mode: host``.

    Behaviour is OS-aware via
    :func:`openfinai_harbor.verifier.client.resolve_url_mode`:

    * Linux (``host`` mode) — write the override if no compose file
      exists. Lets the container share the host's net namespace so
      ``127.0.0.1:<port>`` inside the container IS the host loopback
      (needed on rootless podman and plain Linux docker, where
      ``host.docker.internal`` is not defined).
    * Docker Desktop / Win+Mac (``rewrite`` mode) — DO NOT write the
      override. On Docker Desktop ``network_mode: host`` shares the
      *VM's* loopback rather than the host's, so it doesn't help and
      actively breaks the ``host.docker.internal`` rewrite path. If a
      previously auto-generated override is sitting in the task dir
      (legacy from when ``host`` was the unconditional default), remove
      it so we fall back to bridge networking.

    Hand-authored compose overrides (anything without our auto-gen
    marker comment in the header) are always left alone — task authors
    get the last word.
    """
    from openfinai_harbor.verifier.client import resolve_url_mode

    compose_path = task_dir / "environment" / "docker-compose.yaml"
    mode = resolve_url_mode()

    if mode == "host":
        if compose_path.exists():
            return
        compose_yaml = (
            f"# {_AUTO_COMPOSE_MARKER}.\n"
            "#\n"
            "# network_mode: host lets the agent container reach the\n"
            "# host verifier RPC at 127.0.0.1:<port> on rootless podman\n"
            "# / plain Linux docker (where the slirp4netns or bridge\n"
            "# gateway is otherwise NOT directly routable, and\n"
            "# host.docker.internal is not defined).\n"
            "services:\n"
            "  main:\n"
            "    network_mode: host\n"
        )
        compose_path.write_text(compose_yaml, encoding="utf-8")
        return

    # mode == 'rewrite' (Docker Desktop): drop a stale auto-generated
    # override, but never touch a hand-authored one.
    if not compose_path.exists():
        return
    try:
        head = compose_path.read_text(encoding="utf-8")
    except OSError:
        return
    if _AUTO_COMPOSE_MARKER in head:
        compose_path.unlink()


def _load_environment_overrides(config_path: str | None) -> dict | None:
    """Return the ``environment`` block of a harbor trial-config yaml, or None.

    A non-None result switches the trial off the default Docker environment
    and onto whatever the yaml declares (e.g. the pre-built-SIF
    ``PrebuiltSifSingularityEnvironment`` via ``import_path``). Only the
    ``environment`` section is consumed — the agent / verifier / trials_dir
    are owned by ``run_trial``'s own CLI flags.

    Resolution order: explicit ``--harbor-trial-config`` then the
    ``HARBOR_TRIAL_CONFIG`` env var, so eval/baseline launchers can select the
    Singularity env exactly the way the RL entrypoint does (one source of
    truth). Returns None (→ Docker) when neither is set.
    """
    path = config_path or os.environ.get("HARBOR_TRIAL_CONFIG")
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        print(
            f"[run_trial] HARBOR_TRIAL_CONFIG not found: {p} — using Docker env",
            file=sys.stderr,
        )
        return None
    import yaml

    full = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    env_section = full.get("environment")
    return dict(env_section) if isinstance(env_section, dict) else None


def _build_environment_config(
    *,
    env_overrides: dict | None,
    task_mounts: list,
    container_env: dict,
    force_build: bool,
    delete: bool,
):
    """Build the trial's ``EnvironmentConfig`` for either Docker or a yaml env.

    ``delete`` maps to ``EnvironmentConfig.delete``. We default it to False
    (see the call site) because Harbor's ``delete=True`` teardown runs
    ``docker compose down --rmi all``, which removes the shared prebuilt base
    image — forcing an expensive re-pull/rebuild on every launch. False keeps
    the cached base across trials while still tearing down containers. Applied
    to the Docker path only; the custom-env path leaves ``delete`` to its yaml
    (Harbor's default), since Singularity teardown never runs a docker ``--rmi``.

    Docker (``env_overrides is None``): the legacy path — bind the per-task
    ``/data`` mount and the verifier env into a ``DOCKER`` environment.

    Custom (``env_overrides`` from a yaml): start from the yaml's
    ``environment`` block (carrying ``import_path``/overrides/``kwargs`` for
    e.g. Singularity) and merge the two pieces ``run_trial`` computes at
    runtime — the per-task ``/data`` bind is appended to any yaml mounts
    (e.g. the conda bind), and the verifier coordinates (``VERIFIER_URL`` /
    ``VERIFIER_TOKEN`` / ``AGENT_ID`` / ``HARBOR_TRIAL_DIR``) are merged on top
    of the yaml env (which typically pins ``PATH``/``LD_LIBRARY_PATH``).
    """
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import EnvironmentConfig

    if env_overrides is None:
        return EnvironmentConfig(
            type=EnvironmentType.DOCKER,
            force_build=force_build,
            delete=delete,
            mounts=task_mounts or None,
            env=container_env,
        )

    env_dict = dict(env_overrides)
    yaml_mounts = list(env_dict.get("mounts") or [])
    task_mount_dicts = [
        (m.model_dump() if hasattr(m, "model_dump") else dict(m))
        for m in task_mounts
    ]
    env_dict["mounts"] = (yaml_mounts + task_mount_dicts) or None
    merged_env = dict(env_dict.get("env") or {})
    merged_env.update(container_env)
    env_dict["env"] = merged_env
    env_dict["force_build"] = force_build
    return EnvironmentConfig(**env_dict)


async def _async_main(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    task_dir = Path(args.task_dir).resolve()
    if not task_dir.exists():
        print(f"[run_trial] task_dir not found: {task_dir}", file=sys.stderr)
        return 2

    # Default trials dir: ``data/run_output/examples/<task_name>/``.
    # Override via ``--trials-dir`` (or its ``--output-dir`` alias).
    if args.trials_dir is not None:
        trials_dir = Path(args.trials_dir).resolve()
    else:
        trials_dir = (
            repo_root / "data" / "run_output" / "examples" / task_dir.name
        ).resolve()
    trials_dir.mkdir(parents=True, exist_ok=True)

    # A custom environment (e.g. Singularity on a SLURM cluster) is selected by
    # --harbor-trial-config / $HARBOR_TRIAL_CONFIG. It skips the Docker-only
    # base-image build + host-network compose shim (Singularity shares the
    # host net namespace natively, and there's no Docker daemon on HPC).
    env_overrides = _load_environment_overrides(args.harbor_trial_config)
    use_custom_env = env_overrides is not None
    if not use_custom_env:
        _ensure_host_network_compose(task_dir)

    if args.system_prompt_path is not None:
        sp = Path(args.system_prompt_path)
        if not sp.is_file():
            print(
                f"[run_trial] --system-prompt-path not found: {sp}",
                file=sys.stderr,
            )
            return 2

    # Late imports — relies on the venv with harbor + openfinai_harbor installed.
    from harbor import Trial
    from harbor.models.trial.config import (
        AgentConfig,
        TaskConfig,
        TrialConfig,
        VerifierConfig,
    )
    from openfinai_harbor.docker_base import ensure_base_image
    from openfinai_harbor.runner import (
        build_container_env,
        build_task_data_mount,
        inspect_curated_task,
        maybe_emit_market_closed_warning,
        preflight_curated_credentials,
        pretty_scores,
    )
    from openfinai_harbor.verifier import client as verifier_client
    from openfinai_harbor.verifier.client import container_url_for

    # Ensure the configured registry image is available on Docker paths.
    if not use_custom_env:
        ensure_base_image(
            dockerfile=repo_root / "docker" / "Dockerfile.base",
            build_context=repo_root,
            force_rebuild=args.rebuild_base_image,
            log_prefix="[run_trial]",
        )

    plan = inspect_curated_task(task_dir)
    cred_err = preflight_curated_credentials(plan)
    if cred_err:
        print(f"[run_trial] {cred_err}", file=sys.stderr)
        return 5

    # 1. Spawn / attach the host verifier
    endpoint = None
    aid = f"run-trial-{uuid.uuid4()}"
    container_url = ""
    if plan.skip_rpc_verifier:
        print(
            f"[run_trial] family={plan.family!r} skips host RPC verifier; "
            "running with VERIFIER_URL unset (in-container scoring).",
            flush=True,
        )
    else:
        print("[run_trial] spawning / attaching host verifier ...", flush=True)
        # A per-job registry prevents shared filesystems from mixing node-local
        # verifier endpoints. Unset retains the single-run default.
        _vroot = os.environ.get("OPENFINAI_VERIFIER_RUN_OUTPUT_ROOT")
        try:
            endpoint = verifier_client.get_or_spawn(
                task_dir,
                despawn_grace_min=args.despawn_grace_min,
                bind=args.verifier_bind,
                max_requests_per_minute=args.max_requests_per_minute,
                run_output_root=Path(_vroot) if _vroot else None,
            )
        except Exception as exc:
            print(f"[run_trial] verifier spawn failed: {exc}", file=sys.stderr)
            return 4
        print(
            f"[run_trial] verifier endpoint url={endpoint.url} pid={endpoint.pid} "
            f"task_id={endpoint.task_id}",
            flush=True,
        )
        aid, _ = verifier_client.register(
            endpoint, aid, notes="openfinai_harbor.run_trial"
        )
        container_url = container_url_for(endpoint.url)

    # Compute trial_dir before build_container_env so HARBOR_TRIAL_DIR
    # is plumbed in; submit helpers stamp it as a back-pointer on each
    # ledger row for post-trial inspector/orphan-detection tooling.
    trial_name = _utc_trial_name()
    trial_dir = trials_dir / trial_name
    container_env = build_container_env(
        plan=plan,
        container_url=container_url,
        token=endpoint.token if endpoint is not None else "",
        agent_id=aid,
        trial_dir=trial_dir,
    )

    # 2. Build agent kwargs
    agent_kwargs: Dict[str, Any] = _parse_kv(args.ak or [])
    # Top-level convenience flags map onto the common BaseFinAgent kwargs;
    # explicit --ak values win so callers can always override.
    if args.max_tokens is not None and "max_tokens" not in agent_kwargs:
        agent_kwargs["max_tokens"] = args.max_tokens
    if args.temperature is not None and "temperature" not in agent_kwargs:
        agent_kwargs["temperature"] = args.temperature
    if args.system_prompt_path is not None and "system_prompt_path" not in agent_kwargs:
        agent_kwargs["system_prompt_path"] = str(Path(args.system_prompt_path).resolve())

    # 3. Build TrialConfig
    # Expose task runtime assets at /data.
    task_mounts = [m for m in (build_task_data_mount(task_dir),) if m]
    # EnvironmentConfig.env reaches both the agent and verifier subprocess;
    # VerifierConfig silently ignores environment fields in this Harbor version.
    cfg = TrialConfig(
        task=TaskConfig(path=task_dir),
        trial_name=trial_name,
        trials_dir=trials_dir,
        agent=AgentConfig(
            import_path=args.agent_import_path,
            model_name=_format_provider_model_id(args.provider or "", args.model),
            kwargs=agent_kwargs,
            override_timeout_sec=float(args.timeout_sec),
        ),
        environment=_build_environment_config(
            env_overrides=env_overrides,
            task_mounts=task_mounts,
            container_env=container_env,
            force_build=args.rebuild_image,
            delete=args.cleanup_images,
        ),
        verifier=VerifierConfig(),
    )

    print(f"[run_trial] trial_name = {cfg.trial_name}", flush=True)
    print(f"[run_trial] trial_dir  = {trial_dir}", flush=True)

    # 4. Run
    trial = await Trial.create(cfg)

    async def _heartbeat_loop():
        """Ping ``/heartbeat`` while the trial is running.

        Host verifier's idle-shutdown timer (default 120 s between any
        inbound request) trips even on short multi-turn agent loops
        that take >2 min between requests — a frequent case with Claude
        extended thinking. Without these pings the despawned verifier
        breaks the verifier-step's ``submit_and_persist`` POST with
        Connection Refused. Interval is one-third of the despawn grace
        (capped at 60 s) so at least three pings land before timeout.
        """
        if endpoint is None:
            return
        interval = max(15.0, min(60.0, args.despawn_grace_min * 60.0 / 3.0))
        while True:
            try:
                await asyncio.sleep(interval)
                verifier_client.heartbeat(endpoint, aid)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                print(
                    f"[run_trial] heartbeat warning (verifier may be down): {exc}",
                    file=sys.stderr,
                )

    # Rich Live wrapper mirrors harbor.Job's two-row pattern. Progress
    # widgets are intentionally WITHOUT console=: Live owns the rendering
    # surface, an explicit console would fight it. refresh_per_second=10
    # matches harbor.Job.
    from harbor.trial.hooks import TrialEvent  # noqa: E402

    loading_progress = Progress(
        SpinnerColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    running_progress = Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        TextColumn("[progress.description]{task.description}"),
    )

    # Add tasks BEFORE Live so the first frame has rows; add_task inside
    # the `with` block leaves the first frame empty and the widget may be
    # missed if a hook fires before the next refresh.
    load_task_id = loading_progress.add_task("Running trials...", total=1)
    stage_task_id = running_progress.add_task(
        f"{cfg.trial_name}: starting"
    )

    def _stage(description: str) -> None:
        running_progress.update(stage_task_id, description=description)

    async def _on_env_start(_evt: Any) -> None:
        _stage(f"{cfg.trial_name}: environment setup")

    async def _on_agent_start(_evt: Any) -> None:
        _stage(f"{cfg.trial_name}: running agent")

    async def _on_verify_start(_evt: Any) -> None:
        _stage(f"{cfg.trial_name}: scoring")

    async def _on_end(_evt: Any) -> None:
        _stage(f"{cfg.trial_name}: done")
        loading_progress.update(load_task_id, advance=1)

    trial.add_hook(TrialEvent.ENVIRONMENT_START, _on_env_start)
    trial.add_hook(TrialEvent.AGENT_START, _on_agent_start)
    trial.add_hook(TrialEvent.VERIFICATION_START, _on_verify_start)
    trial.add_hook(TrialEvent.END, _on_end)

    with Live(
        Group(loading_progress, running_progress),
        refresh_per_second=10,
    ):
        hb_task = asyncio.create_task(_heartbeat_loop())
        try:
            result = await trial.run()
        finally:
            hb_task.cancel()
            try:
                await hb_task
            except (asyncio.CancelledError, Exception):
                pass
            if endpoint is not None:
                try:
                    verifier_client.deregister(endpoint, aid)
                except Exception:
                    pass

    rewards = (
        result.verifier_result.rewards
        if result.verifier_result is not None
        else None
    )
    maybe_emit_market_closed_warning(rewards, trial_dir, log_prefix="[run_trial]")
    _render_trial_summary(
        result=result,
        trial_name=cfg.trial_name,
        trial_dir=trial_dir,
        rewards=rewards,
    )
    # stderr line kept for log-grep compatibility; the Rich table is
    # the human-facing surface.
    print(f"[run_trial] scores: {pretty_scores(rewards)}", file=sys.stderr)
    if result.exception_info is not None:
        return 1
    if rewards is None:
        return 1
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m openfinai_harbor.run_trial",
        description=(
            "Pre-spawn the host verifier RPC, then run a single Harbor trial. "
            "Required for auto-pipeline / converted-curated task bundles "
            "because `harbor run` alone does not spawn the host verifier."
        ),
    )
    p.add_argument(
        "--task-dir",
        required=True,
        help="Path to the installed task bundle",
    )
    p.add_argument(
        "--agent-import-path",
        default="openfinai_harbor.agents:SingleShotLLMAgent",
        help="`module:Class` of the agent (default: SingleShotLLMAgent)",
    )
    p.add_argument(
        "--model",
        required=True,
        help=(
            "LLM model identifier. Either a bare name (then pair with "
            "--provider) or a fully-qualified `provider/model`."
        ),
    )
    p.add_argument(
        "--provider",
        default=None,
        help="LiteLLM provider (e.g. ollama, openrouter, anthropic). "
        "Ignored when --model already carries a `provider/` prefix.",
    )
    p.add_argument(
        "--trials-dir",
        "--output-dir",
        default=None,
        help=(
            "Output dir. Defaults to "
            "data/run_output/examples/<task_name>/ when unset. "
            "``--output-dir`` is kept as an alias for legacy invocations."
        ),
    )
    p.add_argument(
        "--timeout-sec",
        type=float,
        default=600.0,
        help="Per-trial timeout (default: 600)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Convenience: agent max_tokens kwarg (overridable via --ak).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Convenience: agent temperature kwarg (overridable via --ak).",
    )
    p.add_argument(
        "--system-prompt-path",
        "--system-prompt-file",
        default=None,
        help=(
            "Convenience: agent system_prompt_path kwarg (overridable via "
            "--ak). Pre-flighted for existence so a typo fails here, not "
            "deep inside the agent's __init__. "
            "``--system-prompt-file`` is kept as an alias for legacy "
            "invocations."
        ),
    )
    p.add_argument(
        "--ak",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra agent kwargs (repeatable). Mirrors harbor CLI's --ak "
            "convention. e.g. --ak max_turns=5 --ak temperature=0.0. "
            "Values are auto-coerced: true/false → bool, then int, then "
            "float, otherwise str."
        ),
    )
    p.add_argument(
        "--harbor-trial-config",
        default=None,
        help=(
            "Path to a harbor trial-config yaml whose `environment` block "
            "selects a non-Docker environment (e.g. the Singularity "
            "config). Only the `environment` section is used; agent/verifier "
            "stay CLI-driven. Falls back to $HARBOR_TRIAL_CONFIG. Unset → "
            "Docker (default)."
        ),
    )
    p.add_argument(
        "--rebuild-image",
        action="store_true",
        help="Force rebuild the per-task image",
    )
    p.add_argument(
        "--cleanup-images",
        action="store_true",
        help=(
            "Tear down the Docker environment with ``delete=True`` "
            "(``docker compose down --rmi all --volumes``), removing the "
            "shared base image to reclaim disk. Default (off) keeps the "
            "cached base across trials so it isn't re-pulled every launch."
        ),
    )
    p.add_argument(
        "--rebuild-base-image",
        "--rebuild-base",
        action="store_true",
        help=(
            "Force rebuild openfinai-base. ``--rebuild-base`` is kept as "
            "an alias for legacy invocations."
        ),
    )
    p.add_argument(
        "--despawn-grace-min",
        type=float,
        default=2.0,
        help=(
            "Minutes the verifier keeps running after this run deregisters. "
            "0 = immediate shutdown; 2 leaves a small window so a follow-up "
            "trial on the same task warm-attaches. Default: 2."
        ),
    )
    p.add_argument(
        "--verifier-bind",
        default="127.0.0.1",
        help="Bind interface for the host verifier process (default 127.0.0.1).",
    )
    p.add_argument(
        "--max-requests-per-minute",
        type=int,
        default=None,
        help=(
            "Override the verifier's per-token rate limit (default 60/min). "
            "Set high (e.g. 100000) for RL training that submits at high "
            "frequency; unset for typical single-trial runs."
        ),
    )
    return p.parse_args()


def main() -> int:
    return asyncio.run(_async_main(_parse_args()))


if __name__ == "__main__":
    sys.exit(main())
