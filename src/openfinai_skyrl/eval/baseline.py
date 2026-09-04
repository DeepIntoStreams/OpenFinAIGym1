"""Zero-shot baseline evaluator: same flow as eval.trained, but without any adapter."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from openfinai_skyrl.eval.common import (
    AGENT_BY_MODE,
    resolve_test_tasks,
    system_prompt_from_dataset_manifest,
)
from openfinai_skyrl.eval.trained import run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["single", "multi"], required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks-root", default="tasks")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-tasks", nargs="*", default=None)
    parser.add_argument("--system-prompt-path", default=None)
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Prepared dataset path; when --system-prompt-path is omitted, "
        "auto-reads system_prompt_path from <dataset-dir>/dataset_manifest.json "
        "so the baseline pins the same prompt SFT trained on.",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    parser.add_argument("--ak", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--n-seeds", type=int, default=1,
        help="Trials per task on the base LLM. Use the same value as eval-trained "
        "so the (baseline mean ± std) vs (trained mean ± std) comparison is symmetric.",
    )
    parser.add_argument(
        "--seed-base", type=int, default=0,
        help="First seed (incremented for each of --n-seeds replicas).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_import_path = AGENT_BY_MODE[args.mode]
    tasks_root = Path(args.tasks_root).resolve()
    test_tasks = resolve_test_tasks(args.test_tasks, tasks_root)
    output_dir = Path(args.output_dir).resolve()
    system_prompt_path = args.system_prompt_path or system_prompt_from_dataset_manifest(
        args.dataset_dir
    )
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
