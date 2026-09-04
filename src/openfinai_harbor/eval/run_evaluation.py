#!/usr/bin/env python3
"""
Example evaluation entry-point executed **inside** the container by test.sh.

Usage (from test.sh):
    python /openfinai/eval/run_evaluation.py \
        --task crypto_forecasting/my_task \
        --data-dir /data \
        --reward-output /logs/verifier/reward.json

This script:
1. Locates the assembled evaluator for the given task name.
2. Locates the agent's generated code (e.g. /workspace/train.py).
3. Runs it inside the benchmark data directory.
4. Loads the agent's predictions and the ground-truth data.
5. Passes them to the assembled evaluator which computes reward scores,
   prints a summary, and writes reward.json.
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


def _find_evaluator(task_name: str) -> Path:
    """Locate the assembled evaluator.py for the given task name.

    Search order:
    1. Container path: ``/openfinai/eval/evaluator.py``
    2. Benchmark install: ``<repo>/tasks/generated/<scope>/<task>/evaluator.py``
       *task_name* may be ``<scope>/<task_id>`` or just ``<task_id>`` (scans all scopes).

    Returns the first match found.
    """
    container_path = Path("/openfinai/eval/evaluator.py")
    if container_path.is_file():
        return container_path

    repo_root = Path(__file__).resolve().parents[4]
    benchmark_root = repo_root / "tasks" / "generated"

    if "/" in task_name:
        candidate = benchmark_root / task_name / "evaluator.py"
        if candidate.is_file():
            return candidate
    else:
        if benchmark_root.is_dir():
            for scope_dir in sorted(benchmark_root.iterdir()):
                if not scope_dir.is_dir():
                    continue
                candidate = scope_dir / task_name / "evaluator.py"
                if candidate.is_file():
                    return candidate

    raise FileNotFoundError(
        f"No evaluator.py found for task '{task_name}'. "
        f"Searched: {container_path}, {benchmark_root}/<scope>/{task_name}/evaluator.py"
    )


def _load_evaluator(evaluator_path: Path) -> Any:
    """Dynamically import the assembled evaluator and return an instance.

    The evaluator module adds the reward bank to ``sys.path`` itself, but
    ``openfinai_pipeline`` must already be importable (set PYTHONPATH in the
    Dockerfile or test.sh if needed).
    """
    spec = importlib.util.spec_from_file_location("evaluator", str(evaluator_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator from {evaluator_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if (
            isinstance(obj, type)
            and hasattr(obj, "REWARD_NAMES")
            and hasattr(obj, "score")
            and obj.__name__ != "BaseEvaluator"
        ):
            return obj()
    raise RuntimeError(f"No evaluator class found in {evaluator_path}")


def _load_array(directory: Path, *candidates: str) -> Optional[np.ndarray]:
    """Try loading an array from the first matching file in *directory*."""
    for name in candidates:
        path = directory / name
        if not path.exists():
            continue
        if path.suffix == ".npy":
            return np.load(str(path))
        if path.suffix == ".npz":
            data = np.load(str(path))
            return data[list(data.files)[0]]
        if path.suffix == ".csv":
            return np.loadtxt(str(path), delimiter=",", skiprows=1, encoding="utf-8")
    return None


def _load_predictions(work_dir: Path) -> Optional[np.ndarray]:
    """Load agent predictions from the working directory."""
    return _load_array(
        work_dir,
        "predictions.npy",
        "predictions.npz",
        "predictions.csv",
        "output.npy",
        "output.csv",
        "results.npy",
        "results.csv",
    )


def _load_ground_truth(data_dir: Path) -> Optional[np.ndarray]:
    """Load ground truth from the benchmark data directory."""
    return _load_array(
        data_dir,
        "ground_truth.npy",
        "ground_truth.npz",
        "ground_truth.csv",
        "test_labels.npy",
        "test_labels.csv",
        "y_test.npy",
        "y_test.csv",
    )


def _find_train_py() -> Optional[Path]:
    """Look for the agent-generated train.py in common locations."""
    candidates = [
        Path("/workspace/train.py"),
        Path("/workspace/env/train.py"),
        Path("/logs/agent/train.py"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _extract_train_code_from_conversation() -> Optional[str]:
    """Fallback: extract code from the agent conversation log."""
    conv_path = Path("/logs/agent/conversation.json")
    if not conv_path.exists():
        return None
    try:
        conversation = json.loads(conv_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    import re

    candidates = []
    for msg in reversed(conversation):
        if msg.get("role") != "assistant":
            continue
        text = msg.get("content", "")
        for pat in (r"<python>(.*?)</python>", r"```python(.*?)```"):
            for m in re.finditer(pat, text, re.DOTALL):
                code = m.group(1).strip()
                if code and (
                    "train_model" in code or "modify_config" in code or len(code) > 200
                ):
                    candidates.append(code)
    return max(candidates, key=len) if candidates else None


def _run_train(
    code_or_path: str | Path,
    data_dir: Path,
    timeout: int = 600,
) -> Tuple[Optional[Path], bool]:
    """Run train.py in a copy of *data_dir*.

    Returns ``(work_dir, success)``.  The caller is responsible for
    cleaning up *work_dir* after loading any output files it needs.
    """
    work = tempfile.mkdtemp(prefix="openfinai_eval_")
    work_path = Path(work)
    try:
        for item in data_dir.iterdir():
            src = data_dir / item.name
            dst = work_path / item.name
            if src.is_dir():
                shutil.copytree(str(src), str(dst), symlinks=True)
            else:
                shutil.copy2(str(src), str(dst))

        train_py = work_path / "train.py"
        if isinstance(code_or_path, Path) and code_or_path.is_file():
            shutil.copy2(str(code_or_path), str(train_py))
        else:
            train_py.write_text(str(code_or_path), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_path),
        )

        if result.returncode != 0:
            print(
                f"[evaluate] train.py failed:\n{result.stderr[:2000]}", file=sys.stderr
            )
            return work_path, False

        return work_path, True

    except subprocess.TimeoutExpired:
        print(f"[evaluate] train.py timed out after {timeout}s", file=sys.stderr)
        return work_path, False
    except Exception as exc:
        print(f"[evaluate] error: {exc}", file=sys.stderr)
        return work_path, False


def _write_zero_reward(
    reward_names: list[str],
    reward_path: Path,
    error: str,
) -> None:
    """Write a reward.json with all-zero scores and an error message."""
    scores = {name: 0.0 for name in reward_names}
    scores["reward"] = 0.0
    scores["_error"] = error
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(json.dumps(scores, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenFinGym evaluation runner")
    parser.add_argument(
        "--task",
        required=True,
        help="Task name — either '<scope>/<task_id>' or bare '<task_id>'",
    )
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--weights",
        default=None,
        help="Comma-separated reward weights following REWARD_NAMES order",
    )
    parser.add_argument(
        "--reward-output",
        default="/logs/verifier/reward.json",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    reward_path = Path(args.reward_output)

    # Discover and load the assembled evaluator for this task
    evaluator_path = _find_evaluator(args.task)
    evaluator = _load_evaluator(evaluator_path)
    print(
        f"[evaluate] Loaded evaluator: {evaluator.__class__.__name__} ({evaluator_path})"
    )
    print(f"[evaluate] Reward functions: {evaluator.REWARD_NAMES}")

    # Parse weights
    weights: list[float] | None = None
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]

    # Locate agent code
    train_path = _find_train_py()
    code_source: str | Path | None = train_path
    if code_source is None:
        code_source = _extract_train_code_from_conversation()

    if code_source is None:
        _write_zero_reward(
            evaluator.REWARD_NAMES, reward_path, "No train.py or extractable code found"
        )
        print("[evaluate] No code found — reward = 0")
        return

    # Run agent code
    print(f"[evaluate] Running agent code (timeout={args.timeout}s) ...")
    start = time.time()
    work_dir, success = _run_train(code_source, data_dir, timeout=args.timeout)
    elapsed = time.time() - start
    print(f"[evaluate] Agent finished in {elapsed:.1f}s (success={success})")

    try:
        if not success:
            _write_zero_reward(
                evaluator.REWARD_NAMES, reward_path, "Agent code failed or timed out"
            )
            print("[evaluate] Agent failed — reward = 0")
            return

        # Load predictions (agent output) and ground truth (benchmark data)
        predictions = _load_predictions(work_dir)
        ground_truth = _load_ground_truth(data_dir)

        if predictions is None:
            _write_zero_reward(
                evaluator.REWARD_NAMES,
                reward_path,
                "No predictions file found in agent output",
            )
            print("[evaluate] No predictions found — reward = 0")
            return

        if ground_truth is None:
            _write_zero_reward(
                evaluator.REWARD_NAMES,
                reward_path,
                "No ground truth file found in data directory",
            )
            print("[evaluate] No ground truth found — reward = 0")
            return

        # ``evaluator.score`` persists reward.json itself (via
        # ``reward_output``); the returned dict is echoed for the log.
        scores = evaluator.score(
            predictions=predictions,
            ground_truth=ground_truth,
            weights=weights,
            reward_output=args.reward_output,
        )
        print(f"[evaluate] scores: {scores}")
        print(f"[evaluate] Evaluation complete (elapsed={elapsed:.1f}s)")

    finally:
        if work_dir is not None:
            shutil.rmtree(str(work_dir), ignore_errors=True)


if __name__ == "__main__":
    main()
