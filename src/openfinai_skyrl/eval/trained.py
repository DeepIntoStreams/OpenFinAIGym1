"""End-to-end execution eval: run the trained agent against held-out test tasks via run_trial."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from openfinai_skyrl.eval.common import (
    AGENT_BY_MODE,
    newest_trial_dir,
    read_reward,
    resolve_agent_import_path,
    resolve_test_tasks,
    system_prompt_from_dataset_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["single", "multi"], default=None,
                        help="Agent class; defaults to the run manifest value.")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="SFT run dir; must contain manifest.json.")
    parser.add_argument("--provider", required=True,
                        help="LiteLLM provider (vllm, ollama, openai, ...).")
    parser.add_argument("--model", required=True,
                        help="Model name as registered with the served endpoint.")
    parser.add_argument("--tasks-root", default="tasks")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-tasks", nargs="*", default=None,
                        help="Override the 5 default held-out test tasks.")
    parser.add_argument("--system-prompt-path", default=None,
                        help="Forwarded to the agent via --ak.")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout-sec", type=float, default=1800.0,
                        help="Per-trial wall clock cap.")
    parser.add_argument("--ak", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra agent kwargs (repeatable); forwarded to run_trial.")
    parser.add_argument(
        "--ckpt-steps",
        default=None,
        help=(
            "Comma-separated step numbers or 'all' to sweep "
            "<checkpoint-dir>/checkpoints/step_<N>/; each lands under "
            "--output-dir/by_step/step_<N>/. Default evaluates hf_final/ only."
        ),
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Trials per task; each seed reruns with --ak seed=<S>. Use >=3 for CIs.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="First seed (incremented for each of --n-seeds replicas).",
    )
    parser.add_argument(
        "--auto-merge",
        action="store_true",
        help=(
            "Materialise <checkpoint-dir>/hf_merged/ from base + LoRA adapter. "
            "Operator must point the serving endpoint at hf_merged/ themselves."
        ),
    )
    parser.add_argument(
        "--skip-endpoint-check",
        action="store_true",
        help="Skip the /v1/models sanity check; disable only for endpoints without /v1/models.",
    )
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help=(
            "Base URL for the /v1/models check. Defaults to OPENAI_API_BASE / "
            "LITELLM_BASE_URL / VLLM_BASE_URL / http://127.0.0.1:8000."
        ),
    )
    return parser.parse_args()


def _load_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_endpoint_url(cli_url: str | None) -> str:
    """Pick the base URL for /v1/models; strips trailing /v1 so either form works.

    ``rstrip("/v1")`` would strip any trailing combination of ``/``, ``v``, ``1`` —
    e.g. ``…:8001/v1`` becomes ``…:800``. Use a literal suffix check instead.
    """
    raw = (
        cli_url
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("LITELLM_BASE_URL")
        or os.environ.get("VLLM_BASE_URL")
        or "http://127.0.0.1:8000"
    )
    base = raw.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base.rstrip("/")


def _endpoint_advertises_model(base_url: str, model_id: str, timeout_sec: float = 10.0) -> tuple[bool, str]:
    """Return (ok, detail): ok iff /v1/models lists an entry with id==model_id."""
    try:
        import urllib.request
        import urllib.error
    except Exception as exc:
        return False, f"urllib import failed: {exc}"

    url = f"{base_url}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return False, f"could not reach {url}: {exc}"
    except json.JSONDecodeError as exc:
        return False, f"{url} returned non-JSON: {exc}"

    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return False, f"{url} payload had no 'data' list (got {type(payload).__name__})"
    served_ids = [str(entry.get("id")) for entry in entries if isinstance(entry, dict)]
    if model_id in served_ids:
        return True, f"endpoint serves {model_id!r}"
    return False, (
        f"endpoint at {url} does NOT advertise model_id={model_id!r}. "
        f"Served ids: {served_ids!r}. The eval would silently score whatever "
        f"the endpoint is actually serving (often the base model). Either "
        f"load the adapter into the server, point --model at the correct id, "
        f"or pre-merge with --auto-merge."
    )


def _auto_merge_adapter(checkpoint_dir: Path) -> Path:
    """Materialise ``<checkpoint-dir>/hf_merged/`` from base + LoRA adapter; idempotent.

    For **full-parameter** training (no LoRA), ``hf_final/`` already contains a
    flat HF model and there is nothing to merge — this function detects the
    absence of ``adapter_config.json`` and returns ``hf_final/`` directly so
    callers (and vLLM) can point at one path regardless of training mode.
    """
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"--auto-merge: {manifest_path} not found; cannot determine base model."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_model = manifest.get("model_path")
    # Canonical location is <ckpt_dir>/hf_final; the manifest's ``hf_final_dir``
    # is an absolute path cached at train time and goes stale if the run dir
    # is moved or renamed. Prefer the canonical convention; fall back to the
    # manifest field only when the convention doesn't yield a valid dir.
    adapter_path = checkpoint_dir / "hf_final"
    if not adapter_path.exists():
        manifest_dir = manifest.get("hf_final_dir")
        if manifest_dir and Path(manifest_dir).exists():
            adapter_path = Path(manifest_dir)
    if not base_model:
        raise SystemExit(
            f"--auto-merge: manifest at {manifest_path} has no 'model_path'."
        )
    if not adapter_path.exists():
        raise SystemExit(
            f"--auto-merge: adapter dir {adapter_path} does not exist."
        )
    if not (adapter_path / "adapter_config.json").exists():
        print(
            f"[auto-merge] {adapter_path} has no adapter_config.json — "
            f"treating as a full-parameter checkpoint (already merged). "
            f"Point your serving endpoint at {adapter_path} directly.",
            flush=True,
        )
        return adapter_path
    merged_dir = checkpoint_dir / "hf_merged"
    if any(merged_dir.glob("*.safetensors")) or (merged_dir / "pytorch_model.bin").exists():
        print(f"[auto-merge] reusing existing {merged_dir}", flush=True)
        return merged_dir

    print(
        f"[auto-merge] loading base {base_model!r} + adapter {adapter_path} -> {merged_dir}",
        flush=True,
    )
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            f"--auto-merge requires transformers + peft + torch installed: {exc}"
        )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(base, str(adapter_path))
    merged = merged.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    try:
        AutoTokenizer.from_pretrained(base_model).save_pretrained(str(merged_dir))
    except Exception as exc:
        print(f"[auto-merge] WARN: tokenizer copy failed: {exc}", flush=True)
    print(f"[auto-merge] wrote {merged_dir}", flush=True)
    return merged_dir


def run_eval(
    *,
    agent_import_path: str,
    provider: str,
    model: str,
    tasks_root: Path,
    test_tasks: list[Path],
    output_dir: Path,
    extra_ak: list[str],
    system_prompt_path: str | None,
    max_tokens: int | None,
    temperature: float | None,
    timeout_sec: float,
    n_seeds: int = 1,
    seed_base: int = 0,
) -> dict[str, Any]:
    import statistics

    output_dir.mkdir(parents=True, exist_ok=True)
    trials_root = output_dir / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for task_dir in test_tasks:
        task_trials_dir = trials_root / task_dir.name
        task_trials_dir.mkdir(parents=True, exist_ok=True)
        seed_rewards: list[float] = []
        seed_records: list[dict[str, Any]] = []
        last_returncode = 0
        last_trial_dir: Path | None = None
        last_log_path = output_dir / f"{task_dir.name}.log"

        for k in range(max(1, int(n_seeds))):
            seed = seed_base + k
            cmd = [
                sys.executable,
                "-m",
                "openfinai_harbor.run_trial",
                "--task-dir",
                str(task_dir),
                "--agent-import-path",
                agent_import_path,
                "--provider",
                provider,
                "--model",
                model,
                "--trials-dir",
                str(task_trials_dir),
                "--timeout-sec",
                str(timeout_sec),
            ]
            if max_tokens is not None:
                cmd.extend(["--max-tokens", str(max_tokens)])
            if temperature is not None:
                cmd.extend(["--temperature", str(temperature)])
            if system_prompt_path is not None:
                cmd.extend(["--ak", f"system_prompt_path={system_prompt_path}"])
            if n_seeds > 1:
                cmd.extend(["--ak", f"seed={seed}"])
            for raw in extra_ak:
                cmd.extend(["--ak", raw])

            seed_log = output_dir / f"{task_dir.name}__seed{seed}.log" if n_seeds > 1 else last_log_path
            with seed_log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    cmd, stdout=handle, stderr=subprocess.STDOUT, text=True,
                )
            last_returncode = result.returncode
            trial_dir = newest_trial_dir(task_trials_dir)
            last_trial_dir = trial_dir
            last_log_path = seed_log
            reward = read_reward(trial_dir) if trial_dir is not None else None
            if reward is not None:
                seed_rewards.append(float(reward))
            seed_records.append({
                "seed": seed,
                "trial_dir": str(trial_dir) if trial_dir else None,
                "log_path": str(seed_log),
                "returncode": result.returncode,
                "reward": float(reward) if reward is not None else None,
            })
            print(
                f"[trained] {task_dir.name} seed={seed}: reward={reward} "
                f"returncode={result.returncode}",
                flush=True,
            )

        # Mean/std INCLUDING the verifier failure sentinel (reward=0.0). Kept for
        # backwards-compat: downstream tools still read ``reward`` / ``reward_mean``.
        if seed_rewards:
            reward_mean = float(statistics.mean(seed_rewards))
            reward_std = float(statistics.pstdev(seed_rewards)) if len(seed_rewards) > 1 else 0.0
        else:
            reward_mean = None
            reward_std = None
        # Success-conditional aggregates: the verifier writes ``reward=0`` as a
        # failure sentinel (no train.py, agent code raised, wrong-shape predictions
        # — see ``curated/in_container/run_eval_curated.py``). So the headline
        # signal splits into TWO numbers:
        #   * success_rate         — fraction of seeds where reward>0 (reliability)
        #   * *_success_only       — mean/std restricted to those successes (quality)
        successful_rewards = [r for r in seed_rewards if r > 0]
        n_total = max(1, int(n_seeds))
        n_with_reward = len(seed_rewards)
        n_success = len(successful_rewards)
        success_rate = n_success / n_total
        if successful_rewards:
            reward_mean_success_only = float(statistics.mean(successful_rewards))
            reward_std_success_only = (
                float(statistics.pstdev(successful_rewards))
                if len(successful_rewards) > 1 else 0.0
            )
        else:
            reward_mean_success_only = None
            reward_std_success_only = None
        # Legacy ``success`` flag: True iff at least one seed produced reward>0.
        success = n_success > 0
        row = {
            "task": task_dir.name,
            "task_dir": str(task_dir),
            "trial_dir": str(last_trial_dir) if last_trial_dir else None,
            "returncode": last_returncode,
            "log_path": str(last_log_path),
            "reward": reward_mean,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_n": n_with_reward,
            "reward_n_total": n_total,
            "reward_n_success": n_success,
            "success_rate": success_rate,
            "reward_mean_success_only": reward_mean_success_only,
            "reward_std_success_only": reward_std_success_only,
            "seeds": seed_records,
            "success": bool(success),
        }
        summary_rows.append(row)
        if n_seeds > 1:
            sm_repr = (
                f"{reward_mean_success_only:.4f}"
                if reward_mean_success_only is not None else "None"
            )
            print(
                f"[trained] {task_dir.name}: reward_mean={reward_mean} "
                f"reward_std={reward_std} n={n_with_reward}  "
                f"success_rate={success_rate:.2f} ({n_success}/{n_total})  "
                f"reward_mean_success_only={sm_repr}",
                flush=True,
            )

    rewards = [r["reward"] for r in summary_rows if r.get("reward") is not None]
    success_only_means = [
        r["reward_mean_success_only"]
        for r in summary_rows
        if r.get("reward_mean_success_only") is not None
    ]
    total_attempted = len(summary_rows) * max(1, int(n_seeds))
    total_successes = sum(r["reward_n_success"] for r in summary_rows)
    summary = {
        "agent_import_path": agent_import_path,
        "provider": provider,
        "model": model,
        "tasks_root": str(tasks_root.resolve()),
        "n_tasks": len(summary_rows),
        "n_seeds": int(n_seeds),
        "seed_base": int(seed_base),
        "n_success": sum(1 for r in summary_rows if r["success"]),
        "avg_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
        # New aggregates: report reliability and quality separately.
        "overall_success_rate": (
            total_successes / total_attempted if total_attempted else 0.0
        ),
        "avg_reward_success_only": (
            sum(success_only_means) / len(success_only_means)
            if success_only_means else None
        ),
        "n_tasks_with_any_success": len(success_only_means),
        "tasks": summary_rows,
    }
    # Single JSON output — summary.csv / per_task_reward.{csv,png} were dropped.
    # Per-trial agent + verifier dirs under output_dir/trials/<task>/<utc>/ remain
    # the source of truth; recompute downstream views from those if needed.
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    manifest = _load_manifest(checkpoint_dir)
    agent_import_path = resolve_agent_import_path(args.mode, manifest)
    tasks_root = Path(args.tasks_root).resolve()
    test_tasks = resolve_test_tasks(args.test_tasks, tasks_root)
    output_dir = Path(args.output_dir).resolve()
    # Fall back to the dataset manifest so eval-time matches train-time bytes.
    system_prompt_path = args.system_prompt_path or system_prompt_from_dataset_manifest(
        manifest.get("dataset_name") if isinstance(manifest, dict) else None
    )
    if args.auto_merge:
        merged_dir = _auto_merge_adapter(checkpoint_dir)
        print(
            f"\n[eval/trained] --auto-merge: merged checkpoint at {merged_dir}. "
            "Point your serving endpoint there before running this eval.",
            flush=True,
        )

    # Endpoint sanity check guards against silently scoring the base model.
    bypass_check = (
        args.skip_endpoint_check
        or args.provider.lower() in {"claude-cli", "deterministic", "harbor"}
    )
    if not bypass_check:
        base_url = _resolve_endpoint_url(args.endpoint_url)
        ok, detail = _endpoint_advertises_model(base_url, args.model)
        if not ok:
            raise SystemExit(
                f"[eval/trained] endpoint sanity check FAILED: {detail}\n"
                f"Run with --skip-endpoint-check to bypass (e.g. for endpoints "
                f"without /v1/models), or fix the served model id first."
            )
        print(f"[eval/trained] endpoint sanity OK: {detail}", flush=True)

    print(
        "\n[eval/trained] NOTE: this script does NOT load the SFT checkpoint "
        "into the model server itself -- it only dispatches per-task trials "
        f"against the running endpoint at provider={args.provider!r} "
        f"model={args.model!r}. Confirm the endpoint is serving the trained "
        f"weights from {args.checkpoint_dir!r} (LoRA: adapter merged or "
        "attached; full-param: ckpt loaded). Otherwise this report measures "
        "the BASE model, not your fine-tune.",
        flush=True,
    )
    step_runs: list[tuple[int | None, str, Path]] = []
    if args.ckpt_steps:
        ckpt_root = checkpoint_dir / "checkpoints"
        if args.ckpt_steps.strip().lower() == "all":
            steps = sorted(
                int(p.name.replace("step_", ""))
                for p in ckpt_root.glob("step_*") if p.is_dir()
            )
        else:
            steps = [int(s.strip()) for s in args.ckpt_steps.split(",") if s.strip()]
        for s in steps:
            label = f"step_{s:06d}"
            step_dir = ckpt_root / label
            if not step_dir.exists():
                raise SystemExit(
                    f"--ckpt-steps references {step_dir} but it doesn't exist. "
                    f"Available: {sorted(p.name for p in ckpt_root.glob('step_*'))}"
                )
            step_runs.append((s, label, output_dir / "by_step" / label))
        print(
            f"[trained] per-checkpoint sweep over {len(step_runs)} step(s): "
            f"{[s for s, _, _ in step_runs]}",
            flush=True,
        )

    summaries: list[dict[str, Any]] = []
    if step_runs:
        # Operator must reload each checkpoint into the endpoint between sweep steps.
        for step, label, sub_out in step_runs:
            print(f"\n[trained] --- sweep step={step} ({label}) -> {sub_out} ---", flush=True)
            if not bypass_check:
                base_url = _resolve_endpoint_url(args.endpoint_url)
                ok, detail = _endpoint_advertises_model(base_url, args.model)
                if not ok:
                    raise SystemExit(
                        f"[sweep step={step}] endpoint sanity check FAILED: {detail}\n"
                        "Reload the {label} weights into your endpoint and retry, "
                        "or pass --skip-endpoint-check for automation."
                    )
            s = run_eval(
                agent_import_path=agent_import_path,
                provider=args.provider, model=args.model,
                tasks_root=tasks_root, test_tasks=test_tasks,
                output_dir=sub_out, extra_ak=args.ak,
                system_prompt_path=system_prompt_path,
                max_tokens=args.max_tokens, temperature=args.temperature,
                timeout_sec=args.timeout_sec,
                n_seeds=args.n_seeds, seed_base=args.seed_base,
            )
            s["sweep_step"] = step
            summaries.append(s)
        curve = [
            {
                "step": s["sweep_step"],
                "avg_reward": s.get("avg_reward"),
                "n_tasks": s.get("n_tasks"),
                "n_success": s.get("n_success"),
            }
            for s in summaries
        ]
        (output_dir / "sweep_summary.json").write_text(
            json.dumps({"steps": curve, "per_step": summaries}, indent=2),
            encoding="utf-8",
        )
        print(f"\n[trained] sweep_summary.json -> {output_dir / 'sweep_summary.json'}", flush=True)
        summary = summaries[-1]
    else:
        summary = run_eval(
            agent_import_path=agent_import_path,
            provider=args.provider,
            model=args.model,
            tasks_root=tasks_root,
            test_tasks=test_tasks,
            output_dir=output_dir,
            extra_ak=args.ak,
            system_prompt_path=system_prompt_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
            n_seeds=args.n_seeds,
            seed_base=args.seed_base,
        )
    print(
        f"\nResult:     tasks={summary['n_tasks']} success={summary['n_success']} "
        f"avg_reward={summary['avg_reward']:.4f}",
        flush=True,
    )
    print(f"Output:     {output_dir}", flush=True)


if __name__ == "__main__":
    main()
