#!/usr/bin/env python
"""Aggregate per-model baseline ``summary.json`` files into one result table.

Each ``slurm_baseline.sh`` job writes
``data/run_output/baselines/<served>-<jobid>/summary.json`` (the schema
emitted by ``openfinai_skyrl.eval.trained.run_eval``). This script collects
those, keeps the newest per model, and prints a Markdown table of
``reward_mean ± reward_std (success_rate)`` with rows = test tasks and
columns = models, plus per-model ``avg_reward`` / ``overall_success_rate``.

Usage::

    python scripts/rl/baseline_table.py                 # glob the baselines dir
    python scripts/rl/baseline_table.py DIR_OR_SUMMARY  # explicit paths
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def _find_summaries(args: list[str]) -> list[Path]:
    paths: list[Path] = []
    if args:
        for a in args:
            p = Path(a)
            if p.is_dir():
                paths += [Path(x) for x in glob.glob(str(p / "**" / "summary.json"), recursive=True)]
            elif p.name == "summary.json" and p.exists():
                paths.append(p)
    else:
        paths = [Path(x) for x in glob.glob("data/run_output/baselines/*/summary.json")]
    return paths


def _cell(summary: dict, task: str) -> str:
    for t in summary.get("tasks", []):
        if t.get("task") == task:
            rm, rs, sr = t.get("reward_mean"), t.get("reward_std"), t.get("success_rate")
            if rm is None:
                return "n/a"
            base = f"{rm:.3f}" if rs is None else f"{rm:.3f}±{rs:.3f}"
            return f"{base} ({sr:.0%})" if sr is not None else base
    return "-"


def main() -> int:
    paths = _find_summaries(sys.argv[1:])
    if not paths:
        print("no summary.json found (looked under data/run_output/baselines/*/)")
        return 1

    # Newest summary wins per model id.
    by_model: dict[str, dict] = {}
    for p in sorted(paths, key=lambda x: x.stat().st_mtime):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skip {p}: {exc}", file=sys.stderr)
            continue
        by_model[str(s.get("model", p.parent.name))] = s

    models = list(by_model)
    tasks: list[str] = []
    for s in by_model.values():
        for t in s.get("tasks", []):
            if t.get("task") not in tasks:
                tasks.append(t["task"])

    print(f"# Baseline (no-RL) — {', '.join(models)}\n")
    print("Cell = reward_mean ± std (success_rate over seeds)\n")
    print("| Test task | " + " | ".join(models) + " |")
    print("|" + "---|" * (len(models) + 1))
    for task in tasks:
        print("| " + task + " | " + " | ".join(_cell(by_model[m], task) for m in models) + " |")
    print("| **avg_reward** | " + " | ".join(
        f"{by_model[m].get('avg_reward', 0.0):.3f}" for m in models) + " |")
    print("| **overall_success_rate** | " + " | ".join(
        f"{by_model[m].get('overall_success_rate', 0.0):.0%}" for m in models) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
