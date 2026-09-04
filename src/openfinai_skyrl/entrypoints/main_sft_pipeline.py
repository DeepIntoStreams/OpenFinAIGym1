"""Run configured SFT stages from trajectory collection to reporting.

Stages are ``collect``, ``prepare``, ``eval-baseline``, ``train``,
``auto-merge``, ``eval-trained``, and ``visualize``. Use ``--skip-stages`` or
``--only-stages`` to select a subset. Each run writes a status manifest and
the merged configuration hash.

CLI examples::

    # Full pipeline, single-shot, claude-cli sonnet for collection
    python -m openfinai_skyrl.entrypoints.main_sft_pipeline \\
        --config config/sft.yaml --mode single \\
        --provider claude-cli --model sonnet

    # Re-train + re-eval on an existing collected corpus
    python -m openfinai_skyrl.entrypoints.main_sft_pipeline \\
        --config config/sft.yaml --mode single \\
        --jobs-dir data/run_output/jobs/my_collection \\
        --skip-stages collect,eval-baseline,eval-trained

    # Smoke test: prepare + train only on existing trajectories
    python -m openfinai_skyrl.entrypoints.main_sft_pipeline \\
        --config config/sft.yaml --mode single \\
        --jobs-dir /existing/jobs/dir \\
        --only-stages prepare,train,visualize
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ALL_STAGES: tuple[str, ...] = (
    "collect",
    "prepare",
    "eval-baseline",
    "train",
    "auto-merge",
    "eval-trained",
    "visualize",
)


@dataclass
class StageResult:
    name: str
    started_at: str
    finished_at: str
    elapsed_sec: float
    skipped: bool = False
    skipped_reason: str | None = None
    returncode: int | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    error: str | None = None


# Config + path helpers


def _repo_root() -> Path:
    # __file__ = <repo>/src/openfinai_skyrl/entrypoints/main_sft_pipeline.py
    # parents[0]=entrypoints, [1]=openfinai_skyrl, [2]=src, [3]=<repo>.
    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    """OmegaConf-based yaml loader matching the train-side ``parse_cli`` path."""
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    return raw if isinstance(raw, dict) else {}


def _get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _config_sha256(cfg: dict[str, Any]) -> str:
    import hashlib
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _python_bin() -> str:
    """Pick a sensible python binary — .venv first, then sys.executable."""
    venv = _repo_root() / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Managed vLLM lifecycle for evaluation


def _vllm_log_dir() -> Path:
    return _repo_root() / "logs"


def _kill_gpu_orphans() -> None:
    """SIGKILL every GPU compute process reported by nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        for tok in result.stdout.split():
            tok = tok.strip().rstrip(",")
            if tok.isdigit():
                try:
                    os.kill(int(tok), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


@contextlib.contextmanager
def _managed_vllm(
    model_path: str,
    served_model_name: str,
    *,
    port: int = 8001,
    max_model_len: int = 16384,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.6,
    enforce_eager: bool = True,
    startup_timeout_sec: float = 480.0,
    extra_args: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Spawn a vLLM OpenAI-compat server, yield endpoint info, kill on exit.

    Used by ``--manage-vllm`` to make ``eval-baseline`` / ``eval-trained``
    work end-to-end through the pipeline. Sets ``OPENAI_API_BASE`` and
    ``OPENAI_API_KEY`` in the parent process so harbor's LiteLLM picks them up.

    Assumes the caller's PATH already exposes the conda CUDA toolchain (for
    flashinfer JIT) and rootless podman+conmon (for harbor's containers).
    ``scripts/sft/run_sft_pipeline.sh`` does both when ``MANAGE_VLLM=true``.
    """
    log_dir = _vllm_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"managed_vllm_{served_model_name}.log"

    cmd = [
        _python_bin(),
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", served_model_name,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    # Prepend .venv/bin so the vLLM subprocess finds ninja and other venv tools
    # even when the caller's shell PATH doesn't include the venv.
    venv_bin = str(_repo_root() / ".venv" / "bin")
    env["PATH"] = venv_bin + ":" + env.get("PATH", "")
    env.setdefault("VLLM_HOST_IP", "127.0.0.1")

    print(
        f"[managed-vllm] starting: model={model_path} served_name={served_model_name} "
        f"port={port} max_model_len={max_model_len} log={log_path}",
        flush=True,
    )
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env,
        start_new_session=True,  # own process group so SIGTERM kills children too
    )

    # Save the previous OPENAI_API_BASE/OPENAI_API_KEY so we can restore on exit.
    prev_api_base = os.environ.get("OPENAI_API_BASE")
    prev_api_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = f"http://127.0.0.1:{port}/v1"
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    try:
        # Poll /v1/models until the served name appears.
        url = f"http://127.0.0.1:{port}/v1/models"
        deadline = time.time() + startup_timeout_sec
        last_err: str = ""
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                tail = ""
                try:
                    log_handle.flush()
                    with log_path.open(encoding="utf-8") as f:
                        lines = f.readlines()[-30:]
                        tail = "".join(lines)
                except OSError:
                    pass
                raise RuntimeError(
                    f"[managed-vllm] vLLM exited early with rc={proc.returncode}. "
                    f"Tail of {log_path}:\n{tail}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    ids = [
                        str(item.get("id"))
                        for item in (payload.get("data") or [])
                        if isinstance(item, dict)
                    ]
                    if served_model_name in ids:
                        ready = True
                        elapsed = startup_timeout_sec - (deadline - time.time())
                        print(
                            f"[managed-vllm] ready after {elapsed:.1f}s; "
                            f"endpoint={url} served={ids!r}",
                            flush=True,
                        )
                        break
            except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError) as exc:
                last_err = str(exc)
            time.sleep(2)
        if not ready:
            raise RuntimeError(
                f"[managed-vllm] vLLM did not advertise {served_model_name!r} within "
                f"{startup_timeout_sec}s (last error: {last_err}). See {log_path}."
            )

        yield {
            "port": port,
            "served_model_name": served_model_name,
            "endpoint": f"http://127.0.0.1:{port}/v1",
            "log": str(log_path),
            "pid": proc.pid,
        }
    finally:
        print(f"[managed-vllm] tearing down pid={proc.pid}", flush=True)
        try:
            # Terminate the whole process group (vLLM forks an EngineCore worker).
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            log_handle.close()
            _kill_gpu_orphans()
            time.sleep(2)
            # Restore caller's env so subsequent stages aren't surprised.
            if prev_api_base is None:
                os.environ.pop("OPENAI_API_BASE", None)
            else:
                os.environ["OPENAI_API_BASE"] = prev_api_base
            if prev_api_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = prev_api_key


def _resolve_vllm_kwargs(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """CLI > yaml.vllm > defaults."""
    return {
        "port": int(args.vllm_port if args.vllm_port is not None else _get(cfg, "vllm.port", 8001)),
        "max_model_len": int(
            args.vllm_max_model_len if args.vllm_max_model_len is not None
            else _get(cfg, "vllm.max_model_len", 16384)
        ),
        "dtype": str(_get(cfg, "vllm.dtype", "bfloat16")),
        "gpu_memory_utilization": float(_get(cfg, "vllm.gpu_memory_utilization", 0.6)),
        "enforce_eager": bool(_get(cfg, "vllm.enforce_eager", True)),
    }


# Stage runners — each returns a (returncode, outputs_dict)


def _stage_collect(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[int, dict[str, str]]:
    """Drive ``scripts/sft/collect_trajectories.sh`` once with the supplied env."""
    env = os.environ.copy()
    env["MODE"] = args.mode
    if args.model is not None:
        env["MODEL"] = args.model
    if args.agent_import_path is not None:
        env["AGENT_IMPORT_PATH"] = args.agent_import_path
    if args.n_attempts is not None:
        env["N_ATTEMPTS"] = str(args.n_attempts)
    if args.n_concurrent is not None:
        env["N_CONCURRENT"] = str(args.n_concurrent)
    env["TRIALS_DIR"] = str(args.jobs_dir)
    env["PYTHON_BIN"] = _python_bin()
    script = _repo_root() / "scripts" / "sft" / "collect_trajectories.sh"
    if not script.exists():
        raise FileNotFoundError(f"collect script not found: {script}")
    rc = subprocess.run(["bash", str(script)], env=env, cwd=_repo_root()).returncode
    return rc, {"jobs_dir": str(args.jobs_dir)}


def _stage_prepare(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[int, dict[str, str]]:
    """Call ``openfinai_skyrl.data.prepare.prepare`` directly (no subprocess)."""
    from openfinai_skyrl.data.prepare import prepare
    result = prepare(
        mode=args.mode,
        jobs_dir=str(args.jobs_dir),
        output_root=str(args.experiment_root),
        tasks_root=str(args.tasks_root),
        system_prompt_path=args.system_prompt_path,
        reward_max=args.reward_max,
        test_tasks=args.test_tasks,
        top_k_lowest_reward=_get(cfg, "top_k_lowest_reward", None) if args.top_k_lowest_reward is None else args.top_k_lowest_reward,
    )
    return 0, {
        "dataset_dir": result["dataset_dir"],
        "manifest_path": result["manifest_path"],
    }


def _stage_eval_baseline(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[int, dict[str, str]]:
    """Subprocess ``eval.baseline`` to score the BASE LLM on held-out tasks.

    When ``--manage-vllm`` is set, the pipeline launches vLLM serving the
    base model (``model.path`` from the yaml), waits for ``/v1/models``, runs
    the subprocess, then tears vLLM down. Otherwise the operator must have
    vLLM already running and pass ``--provider`` / ``--model``.
    """
    if args.manage_vllm:
        base_model = args.model_path or _get(cfg, "model.path")
        if not base_model:
            raise SystemExit("eval-baseline with --manage-vllm: model.path is required.")
        served_name = args.model or f"base_{_safe_name(base_model)}"
        provider = args.provider or "openai"
        vllm_kw = _resolve_vllm_kwargs(cfg, args)
        out_dir = args.experiment_root / "baseline" / f"base_{_safe_name(served_name)}"
        with _managed_vllm(base_model, served_name, **vllm_kw):
            cmd = [
                _python_bin(), "-m", "openfinai_skyrl.eval.baseline",
                "--mode", args.mode,
                "--provider", provider, "--model", served_name,
                "--tasks-root", str(args.tasks_root),
                "--output-dir", str(out_dir),
                "--dataset-dir", str(args.experiment_root / "data"),
                "--n-seeds", str(args.n_seeds),
                "--seed-base", str(args.seed_base),
            ]
            if args.system_prompt_path:
                cmd.extend(["--system-prompt-path", args.system_prompt_path])
            max_tokens = args.max_tokens
            if max_tokens is None:
                cfg_val = _get(cfg, "eval_max_tokens")
                if cfg_val is not None:
                    max_tokens = int(cfg_val)
            if max_tokens is not None:
                cmd.extend(["--max-tokens", str(max_tokens)])
            rc = subprocess.run(cmd, cwd=_repo_root()).returncode
        return rc, {"baseline_dir": str(out_dir), "served_model_name": served_name}

    if not args.provider or not args.model:
        raise SystemExit(
            "eval-baseline requires --provider and --model (or pass --manage-vllm "
            "to let the pipeline launch vLLM itself)."
        )
    out_dir = args.experiment_root / "baseline" / f"base_{_safe_name(args.model)}"
    cmd = [
        _python_bin(), "-m", "openfinai_skyrl.eval.baseline",
        "--mode", args.mode,
        "--provider", args.provider, "--model", args.model,
        "--tasks-root", str(args.tasks_root),
        "--output-dir", str(out_dir),
        "--dataset-dir", str(args.experiment_root / "data"),
        "--n-seeds", str(args.n_seeds),
        "--seed-base", str(args.seed_base),
    ]
    if args.system_prompt_path:
        cmd.extend(["--system-prompt-path", args.system_prompt_path])
    rc = subprocess.run(cmd, cwd=_repo_root()).returncode
    return rc, {"baseline_dir": str(out_dir)}


def _resolve_run_name(cfg: dict[str, Any], args: argparse.Namespace) -> str:
    """Resolve the run name used to name the checkpoint dir.

    Source of truth is ``run_name`` in the yaml (so the operator can vary
    it per model without touching code). Falls back to ``sft_<mode>`` so
    runs launched with no yaml still produce a stable path.
    """
    name = _get(cfg, "run_name") or f"sft_{args.mode}"
    return str(name)


def _ckpt_path(cfg: dict[str, Any], args: argparse.Namespace) -> Path:
    return args.experiment_root / "runs" / _resolve_run_name(cfg, args)


def _stage_train(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[int, dict[str, str]]:
    """Subprocess ``main_sft`` with the same yaml + a few CLI overrides."""
    dataset_dir = args.experiment_root / "data" / f"dataset_{args.mode}_task_split"
    ckpt_path = _ckpt_path(cfg, args)
    cmd = [
        _python_bin(), "-m", "openfinai_skyrl.entrypoints.main_sft",
        "--mode", args.mode,
        "--config", str(args.config),
        f"dataset_name={dataset_dir}",
        f"ckpt_path={ckpt_path}",
    ]
    if args.model_path is not None:
        cmd.append(f"model.path={args.model_path}")
    if args.num_steps is not None:
        cmd.append(f"num_steps={args.num_steps}")
    if args.max_length is not None:
        cmd.append(f"max_length={args.max_length}")
    rc = subprocess.run(cmd, cwd=_repo_root()).returncode
    return rc, {"ckpt_path": str(ckpt_path), "hf_final_dir": str(ckpt_path / "hf_final")}


def _stage_auto_merge(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[int, dict[str, str]]:
    """Materialise ``hf_merged/`` for LoRA runs so vLLM can serve a flat HF model.

    No-op for full-param runs (``hf_final/`` IS the merged model). Always
    surfaces a ``serve_dir`` output so the operator knows which path to feed
    into the vLLM ``--model`` flag before launching the ``eval-trained`` stage.
    """
    from openfinai_skyrl.eval.trained import _auto_merge_adapter

    ckpt_path = _ckpt_path(cfg, args)
    serve_dir = _auto_merge_adapter(ckpt_path)
    is_lora = (ckpt_path / "hf_final" / "adapter_config.json").exists()
    merge_kind = "lora-merged" if is_lora else "full-param"
    return 0, {"serve_dir": str(serve_dir), "merge_kind": merge_kind}


def _stage_eval_trained(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[int, dict[str, str]]:
    """Subprocess ``eval.trained`` against the served SFT'd LLM endpoint.

    With ``--manage-vllm``: the pipeline launches vLLM serving the merged
    checkpoint (``<ckpt>/hf_merged`` for LoRA, ``<ckpt>/hf_final`` for
    full-param; the path comes from the ``auto-merge`` stage's output dir),
    waits for ``/v1/models``, runs the eval subprocess, and tears vLLM down.
    Without it, the operator manages vLLM externally.
    """
    ckpt_path = _ckpt_path(cfg, args)
    out_dir = ckpt_path / "eval"

    # Generation cap: CLI override > yaml ``eval_max_tokens`` > agent default.
    max_tokens = args.max_tokens
    if max_tokens is None:
        cfg_value = _get(cfg, "eval_max_tokens")
        if cfg_value is not None:
            max_tokens = int(cfg_value)

    def _build_cmd(provider: str, model_id: str, skip_check: bool) -> list[str]:
        cmd = [
            _python_bin(), "-m", "openfinai_skyrl.eval.trained",
            "--mode", args.mode,
            "--checkpoint-dir", str(ckpt_path),
            "--provider", provider, "--model", model_id,
            "--tasks-root", str(args.tasks_root),
            "--output-dir", str(out_dir),
            "--n-seeds", str(args.n_seeds),
            "--seed-base", str(args.seed_base),
        ]
        if args.system_prompt_path:
            cmd.extend(["--system-prompt-path", args.system_prompt_path])
        if max_tokens is not None:
            cmd.extend(["--max-tokens", str(max_tokens)])
        if skip_check:
            cmd.append("--skip-endpoint-check")
        return cmd

    if args.manage_vllm:
        # Auto-merge writes its serve_dir into the previous-stage outputs;
        # fall back to <ckpt>/hf_merged then <ckpt>/hf_final so the stage works
        # even if --only-stages skipped auto-merge but the dir exists already.
        serve_dir = ckpt_path / "hf_merged"
        if not serve_dir.exists():
            serve_dir = ckpt_path / "hf_final"
        if not serve_dir.exists():
            raise SystemExit(
                f"eval-trained --manage-vllm: neither {ckpt_path}/hf_merged nor "
                f"{ckpt_path}/hf_final exists. Run 'auto-merge' first."
            )
        served_name = args.model or _resolve_run_name(cfg, args)
        provider = args.provider or "openai"
        vllm_kw = _resolve_vllm_kwargs(cfg, args)
        with _managed_vllm(str(serve_dir), served_name, **vllm_kw):
            # /v1/models was already polled by _managed_vllm; eval/trained.py's
            # own check would be redundant but harmless. Skip it for speed.
            rc = subprocess.run(
                _build_cmd(provider, served_name, skip_check=True),
                cwd=_repo_root(),
            ).returncode
        return rc, {"trained_eval_dir": str(out_dir), "served_model_name": served_name}

    if not args.provider or not args.model:
        raise SystemExit(
            "eval-trained requires --provider and --model (or pass --manage-vllm "
            "to let the pipeline launch vLLM itself)."
        )
    rc = subprocess.run(
        _build_cmd(args.provider, args.model, skip_check=args.skip_endpoint_check),
        cwd=_repo_root(),
    ).returncode
    return rc, {"trained_eval_dir": str(out_dir)}


def _stage_visualize(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[int, dict[str, str]]:
    """Direct Python call into visualize.main()."""
    from openfinai_skyrl.data import visualize

    # visualize.main reads sys.argv — invoke its programmatic guts directly.
    plots_dir = args.experiment_root / "report" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = visualize._collect_variants(args.experiment_root)
    baseline = visualize._collect_baseline(args.experiment_root)
    plots: dict[str, str] = {}
    plots.update(visualize._plot_train_loss(rows, plots_dir))
    plots.update(visualize._plot_per_task_reward(rows, plots_dir))
    plots.update(visualize._plot_training_curve(rows, plots_dir))
    plots.update(visualize._plot_lift_over_base_llm(rows, baseline, plots_dir))
    plots.update(visualize._plot_base_vs_trained_per_task(rows, baseline, plots_dir))
    plots.update(visualize._plot_token_dashboard(rows, baseline, plots_dir))
    manifest = {
        "experiment_root": str(args.experiment_root),
        "n_variants": len(rows),
        "baseline": (baseline or {}).get("variant"),
        "plots": plots,
    }
    (plots_dir / "plots_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0, {"plots_dir": str(plots_dir), "n_plots": str(len(plots))}


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


_STAGE_RUNNERS = {
    "collect": _stage_collect,
    "prepare": _stage_prepare,
    "eval-baseline": _stage_eval_baseline,
    "train": _stage_train,
    "auto-merge": _stage_auto_merge,
    "eval-trained": _stage_eval_trained,
    "visualize": _stage_visualize,
}


# Prerequisite validation


def _validate_prereqs(stage: str, cfg: dict[str, Any], args: argparse.Namespace) -> str | None:
    """Return a human-readable reason to skip ``stage``, or None to proceed."""
    if stage == "prepare":
        if not args.jobs_dir.exists():
            return f"jobs-dir {args.jobs_dir} doesn't exist (run 'collect' first or pass --jobs-dir)"
        return None
    if stage == "train":
        dataset_dir = args.experiment_root / "data" / f"dataset_{args.mode}_task_split"
        if not dataset_dir.exists():
            return f"dataset {dataset_dir} doesn't exist (run 'prepare' first)"
        return None
    if stage in ("auto-merge", "eval-trained"):
        hf_final = _ckpt_path(cfg, args) / "hf_final"
        if not hf_final.exists():
            return f"trained checkpoint {hf_final} doesn't exist (run 'train' first)"
        return None
    return None


# CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="config/sft.yaml",
                   help="Shared yaml (read for defaults).")
    p.add_argument("--mode", choices=["single", "multi"], default="single")
    p.add_argument("--experiment-root", default=None,
                   help="Output root. Defaults to data/run_output/experiments-sft.")
    p.add_argument("--jobs-dir", default=None,
                   help="Harbor jobs root for collect/prepare stages. "
                        "Default: <experiment_root>/jobs/collect.")
    p.add_argument("--tasks-root", default="tasks")
    # Stage selection
    p.add_argument("--skip-stages", default="",
                   help=f"Comma-separated list of stages to skip. Valid: {','.join(ALL_STAGES)}")
    p.add_argument("--only-stages", default="",
                   help="Comma-separated list of stages to RUN (mutually exclusive with --skip-stages).")
    # Collect overrides
    p.add_argument("--model", default=None, help="LLM id passed to collect + eval (e.g. claude-cli/sonnet, vllm-served-id).")
    p.add_argument("--provider", default=None, help="LiteLLM provider for eval stages.")
    p.add_argument("--agent-import-path", default=None,
                   help="Override agent class for collection (default per mode).")
    p.add_argument("--n-attempts", type=int, default=None)
    p.add_argument("--n-concurrent", type=int, default=None)
    # Prepare overrides
    p.add_argument("--system-prompt-path", default=None,
                   help="Override SFT system prompt. Default: read from trajectory.")
    p.add_argument("--reward-max", type=float, default=None,
                   help="Keep rows with reward <= REWARD_MAX (loss-scaled: lower=better).")
    p.add_argument("--test-tasks", nargs="*", default=None)
    p.add_argument("--top-k-lowest-reward", type=int, default=None,
                   help="Per-task lowest-reward filter (overrides yaml top_k_lowest_reward).")
    # Train overrides
    p.add_argument("--model-path", default=None, help="Override model.path from yaml.")
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    # Eval overrides
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Eval-time generation cap. CLI > yaml ``eval_max_tokens`` > agent default.")
    p.add_argument("--skip-endpoint-check", action="store_true")
    # Managed vLLM (end-to-end mode): the pipeline launches vLLM for
    # eval-baseline (base model) and eval-trained (merged checkpoint), then
    # tears it down. Without this flag the operator manages vLLM externally
    # and must pass --provider/--model.
    p.add_argument("--manage-vllm", action="store_true",
                   help="Pipeline owns vLLM lifecycle for eval-baseline + eval-trained.")
    p.add_argument("--vllm-port", type=int, default=None,
                   help="Port for managed vLLM. CLI > yaml ``vllm.port`` > 8001.")
    p.add_argument("--vllm-max-model-len", type=int, default=None,
                   help="Max context window for managed vLLM. CLI > yaml ``vllm.max_model_len`` > 16384.")
    args = p.parse_args(argv)

    args.config = Path(args.config).resolve()
    args.experiment_root = Path(
        args.experiment_root
        or _repo_root() / "data" / "run_output" / "experiments-sft"
    ).resolve()
    args.jobs_dir = Path(
        args.jobs_dir or args.experiment_root / "jobs" / "collect"
    ).resolve()
    args.tasks_root = Path(args.tasks_root).resolve()

    if args.skip_stages and args.only_stages:
        raise SystemExit("--skip-stages and --only-stages are mutually exclusive.")
    return args


def _resolve_stages(args: argparse.Namespace) -> list[str]:
    requested = list(ALL_STAGES)
    if args.only_stages:
        only = {s.strip() for s in args.only_stages.split(",") if s.strip()}
        bad = only - set(ALL_STAGES)
        if bad:
            raise SystemExit(f"Unknown stage(s) in --only-stages: {bad}")
        return [s for s in requested if s in only]
    if args.skip_stages:
        skip = {s.strip() for s in args.skip_stages.split(",") if s.strip()}
        bad = skip - set(ALL_STAGES)
        if bad:
            raise SystemExit(f"Unknown stage(s) in --skip-stages: {bad}")
        return [s for s in requested if s not in skip]
    return requested


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _load_yaml(args.config) if args.config.exists() else {}
    stages_to_run = _resolve_stages(args)
    args.experiment_root.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline] config:         {args.config}", flush=True)
    print(f"[pipeline] mode:           {args.mode}", flush=True)
    print(f"[pipeline] experiment:     {args.experiment_root}", flush=True)
    print(f"[pipeline] jobs-dir:       {args.jobs_dir}", flush=True)
    print(f"[pipeline] tasks-root:     {args.tasks_root}", flush=True)
    print(f"[pipeline] stages:         {', '.join(stages_to_run)}", flush=True)
    print(f"[pipeline] config_sha256:  {_config_sha256(cfg)}", flush=True)
    print(flush=True)

    results: list[StageResult] = []
    overall_rc = 0
    for stage in stages_to_run:
        skip_reason = _validate_prereqs(stage, cfg, args)
        started = _utc_now()
        t0 = time.time()
        if skip_reason:
            print(f"[pipeline] {stage}: SKIPPED — {skip_reason}", flush=True)
            results.append(StageResult(
                name=stage, started_at=started, finished_at=_utc_now(),
                elapsed_sec=0.0, skipped=True, skipped_reason=skip_reason,
            ))
            continue
        print(f"[pipeline] === {stage} ===", flush=True)
        try:
            rc, outputs = _STAGE_RUNNERS[stage](cfg, args)
            err = None
        except SystemExit as exc:
            rc, outputs, err = 2, {}, str(exc)
            print(f"[pipeline] {stage}: SystemExit — {exc}", flush=True, file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            rc, outputs, err = 3, {}, repr(exc)
            print(f"[pipeline] {stage}: exception — {exc!r}", flush=True, file=sys.stderr)
        elapsed = time.time() - t0
        results.append(StageResult(
            name=stage, started_at=started, finished_at=_utc_now(),
            elapsed_sec=elapsed, returncode=rc, outputs=outputs, error=err,
        ))
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"[pipeline] {stage}: {status} ({elapsed:.1f}s)", flush=True)
        if rc != 0:
            overall_rc = rc
            # Fail-fast: a broken upstream stage poisons everything downstream.
            print(f"[pipeline] aborting remaining stages due to {stage} failure",
                  flush=True, file=sys.stderr)
            break

    manifest = {
        "config":          str(args.config),
        "config_sha256":   _config_sha256(cfg),
        "mode":            args.mode,
        "experiment_root": str(args.experiment_root),
        "jobs_dir":        str(args.jobs_dir),
        "tasks_root":      str(args.tasks_root),
        "stages_requested": stages_to_run,
        "stages":          [asdict(r) for r in results],
        "started_at":      results[0].started_at if results else _utc_now(),
        "finished_at":     _utc_now(),
        "overall_returncode": overall_rc,
    }
    (args.experiment_root / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[pipeline] manifest -> {args.experiment_root / 'pipeline_manifest.json'}",
          flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(args)
    return int(manifest["overall_returncode"] or 0)


if __name__ == "__main__":
    sys.exit(main())
