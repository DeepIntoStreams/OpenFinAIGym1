#!/usr/bin/env python3
"""Curated evaluation entry-point executed **inside** the container by test.sh.

Mirrors :mod:`openfinai_harbor.eval.run_evaluation_explicit` but for the
curated bundle layout: there's no ``evaluator.py`` to load locally
because the curated verifier owns scoring via the curated task class.
The runner's only job is:

1. Find ``/workspace/train.py`` (the LLM-generated solver).
2. Set ``HARBOR_REWARD_OUTPUT`` so ``submit_*_and_persist`` knows where
   to write reward.json.
3. Run train.py in a copy of ``/data/``.
4. Confirm reward.json was written (else: zero reward + clear error).

Same exit-mode semantics as the auto-pipe verifier-mode runner: any
failure surfaces a numeric-clean reward.json (so harbor's pydantic
schema passes) plus a sibling ``error.txt`` with the diagnostic text.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


def _find_train_py() -> Optional[Path]:
    train_py = Path("/workspace/train.py")
    return train_py if train_py.is_file() else None


def _numeric_only(scores: dict) -> dict:
    """Filter to keys whose values are int/float (excluding bool).

    Mirrors the auto-pipe runner: harbor's
    ``VerifierResult.rewards: dict[str, float | int] | None`` rejects
    non-numeric values, so ``_error`` strings / ``_errors`` dicts /
    ``_submission_id`` etc. must be stripped before write.
    """
    return {
        k: v
        for k, v in scores.items()
        if not isinstance(v, bool) and isinstance(v, (int, float))
    }


def _write_reward_artifacts(
    scores: dict, reward_path: Path, *, error: str | None = None
) -> None:
    """Write the single-key ``reward.json`` + the multi-key ``metrics.json``
    sidecar (+ ``error.txt`` diagnostic when relevant).

    See run_evaluation_explicit._write_reward_artifacts for the rationale
    behind the split — harbor's ``Verifier`` -> ``Mean.compute`` chain
    requires a single-key ``{"reward": <float>}`` JSON; the rich
    per-channel payload moved to ``metrics.json``.
    """
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    numeric = _numeric_only(scores)

    # metrics.json — multi-key numeric payload.
    metrics_path = reward_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(numeric, indent=2), encoding="utf-8")

    # reward.json — strict single-key {"reward": <float>}.
    reward_value = numeric.get("reward", numeric.get("weighted_total", 0.0))
    try:
        reward_value = float(reward_value)
    except (TypeError, ValueError):
        reward_value = 0.0
    reward_path.write_text(
        json.dumps({"reward": reward_value}, indent=2), encoding="utf-8"
    )

    stripped = {k: v for k, v in scores.items() if k not in numeric}
    if not error and not stripped:
        return

    lines: list[str] = []
    if error:
        lines.append(error)
    if stripped:
        if lines:
            lines.append("")
        lines.append(
            "Non-numeric fields filtered from metrics.json "
            "(harbor.VerifierResult.rewards rejects non-numeric values):"
        )
        lines.append(json.dumps(stripped, indent=2, default=str))
    error_txt = reward_path.parent / "error.txt"
    error_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_train(
    train_path: Path,
    data_dir: Path,
    *,
    timeout: int,
    extra_env: dict[str, str],
) -> tuple[Optional[Path], bool]:
    """Run train.py in a copy of *data_dir*. Returns (work_dir, success)."""
    work = tempfile.mkdtemp(prefix="curated_eval_")
    work_path = Path(work)
    try:
        for item in data_dir.iterdir():
            src = data_dir / item.name
            dst = work_path / item.name
            if src.is_dir():
                shutil.copytree(str(src), str(dst), symlinks=True)
            else:
                shutil.copy2(str(src), str(dst))

        local_train = work_path / "train.py"
        if train_path.is_file():
            shutil.copy2(str(train_path), str(local_train))

        env = os.environ.copy()
        env.update(extra_env)

        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_path),
            env=env,
        )
        if result.returncode != 0:
            sys.stderr.write(
                f"[curated-eval] train.py failed:\n{result.stderr[-2000:]}\n"
            )
            return work_path, False
        # Pass-through stdout for log inspection.
        if result.stdout:
            sys.stdout.write(result.stdout)
        return work_path, True
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"[curated-eval] train.py timed out after {timeout}s\n")
        return work_path, False
    except Exception as exc:
        sys.stderr.write(f"[curated-eval] error running train.py: {exc}\n")
        return work_path, False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenFinGym curated-task evaluation runner"
    )
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--reward-output", default="/logs/verifier/reward.json"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    reward_path = Path(args.reward_output)

    verifier_url = os.environ.get("VERIFIER_URL", "").strip()
    if not verifier_url:
        _write_reward_artifacts(
            {"reward": 0.0},
            reward_path,
            error=(
                "VERIFIER_URL env not set; curated bundles require the "
                "host-side verifier RPC service to be reachable from "
                "inside the container (see "
                "python -m openfinai_harbor.run_trial)."
            ),
        )
        sys.exit(1)

    train_path = _find_train_py()
    if train_path is None:
        _write_reward_artifacts(
            {"reward": 0.0},
            reward_path,
            error="No /workspace/train.py found",
        )
        sys.stderr.write("[curated-eval] no train.py — reward = 0\n")
        return

    print(
        f"[curated-eval] verifier mode: VERIFIER_URL={verifier_url!r}"
    )
    print(f"[curated-eval] running agent code (timeout={args.timeout}s) ...")
    start = time.time()
    work_dir, success = _run_train(
        train_path,
        data_dir,
        timeout=args.timeout,
        extra_env={"HARBOR_REWARD_OUTPUT": str(reward_path)},
    )
    elapsed = time.time() - start
    print(f"[curated-eval] agent finished in {elapsed:.1f}s success={success}")

    try:
        if not success:
            _write_reward_artifacts(
                {"reward": 0.0},
                reward_path,
                error="Agent code failed or timed out",
            )
            return
        if not reward_path.exists():
            _write_reward_artifacts(
                {"reward": 0.0},
                reward_path,
                error=(
                    "Agent did not write reward.json. The agent's "
                    "train.py must call one of the harbor_submit "
                    "submit_*_and_persist helpers so the verifier-side "
                    "score lands at HARBOR_REWARD_OUTPUT."
                ),
            )
            sys.stderr.write(
                "[curated-eval] no reward.json — reward = 0\n"
            )
            return
        print(f"[curated-eval] reward.json present at {reward_path}")
    finally:
        if work_dir is not None:
            shutil.rmtree(str(work_dir), ignore_errors=True)


if __name__ == "__main__":
    main()
