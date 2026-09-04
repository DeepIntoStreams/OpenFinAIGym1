#!/usr/bin/env python3
"""Evaluate an agent solver inside a task container.

With ``VERIFIER_URL``, held-out scoring stays behind the host verifier. Without
it, the compatibility path scores a predictions file against mounted eval data.

Usage (from test.sh)::

    python /data/run_evaluation_explicit.py \\
        --evaluator-path /data/evaluator.py \\
        --data-dir /data \\
        --eval-data-dir /eval-data \\
        --reward-output /logs/verifier/reward.json
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


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


def _evaluator_requires_predictions(evaluator: Any) -> bool:
    return bool(getattr(evaluator, "EXPECTS_PREDICTIONS", True))


def _evaluator_requires_ground_truth(evaluator: Any) -> bool:
    return bool(getattr(evaluator, "EXPECTS_GROUND_TRUTH", True))


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
        "samples.npy",
        "samples.npz",
        "samples.csv",
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
    """Look for the agent-generated train.py in the canonical location."""
    train_py = Path("/workspace/train.py")
    return train_py if train_py.is_file() else None


def _run_train(
    code_or_path: str | Path,
    data_dir: Path,
    timeout: int = 600,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Path], bool]:
    """Run train.py in a copy of *data_dir*.

    Returns ``(work_dir, success)``.  The caller is responsible for
    cleaning up *work_dir* after loading any output files it needs.

    ``extra_env`` is merged into the agent subprocess's environment.
    Used by verifier-mode to set ``HARBOR_REWARD_OUTPUT`` /
    ``VERIFIER_URL`` / ``VERIFIER_TOKEN`` / ``AGENT_ID``.
    """
    import os as _os

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

        env = _os.environ.copy()
        if extra_env:
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


def _numeric_only(scores: dict) -> dict:
    """Return a copy of ``scores`` containing only numeric (int/float) values.

    Harbor's ``VerifierResult.rewards`` is typed
    ``dict[str, float | int] | None``; pydantic rejects strings/dicts/lists
    and the trial crashes during VerifierResult construction. We strip
    non-numeric fields (``_error``, ``_errors``, ``_submission_id``, …)
    before writing reward.json so the schema stays clean. ``bool`` is
    excluded explicitly — it's a subclass of ``int`` but is treated
    semantically as non-numeric in the reward channel.
    """
    return {
        k: v
        for k, v in scores.items()
        if not isinstance(v, bool) and isinstance(v, (int, float))
    }


def _write_reward_artifacts(
    scores: dict,
    reward_path: Path,
    *,
    error: str | None = None,
) -> None:
    """Write the single-key ``reward.json`` + the multi-key ``metrics.json``
    sidecar + an optional ``error.txt`` diagnostic.

    File layout written next to *reward_path*::

        reward.json   {"reward": <float>}   (single-key — Mean.compute-safe)
        metrics.json  {<metric>: <value>, ...} (numeric-only filtered copy
                                                of the input ``scores``)
        error.txt     (only when *error* is set or non-numeric fields
                       were filtered)

    Rationale:

    * Harbor's ``Verifier`` reads ``reward_text_path`` first then falls
      back to ``reward_json_path``. We stopped writing ``reward.txt`` and
      now write a strict single-key ``reward.json`` that
      ``harbor.metrics.mean.Mean.compute`` accepts (requires len==1).
    * The richer per-metric payload moved to ``metrics.json`` so
      downstream tooling can still inspect every reward channel
      (``rmse``, ``mae``, ``smape``, …) without crashing harbor's
      pydantic schema or its Mean aggregator.
    * ``scores`` is still numeric-filtered (no ``_error`` strings /
      ``_errors`` dicts) to mirror what older multi-key ``reward.json``
      consumers expected in metrics.json.
    """
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    numeric = _numeric_only(scores)

    metrics_path = reward_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(numeric, indent=2), encoding="utf-8")

    # Falls back to weighted_total if the evaluator put the headline there.
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

    error_lines: list[str] = []
    if error:
        error_lines.append(error)
    if stripped:
        if error_lines:
            error_lines.append("")
        error_lines.append(
            "Non-numeric fields filtered from metrics.json "
            "(harbor.VerifierResult.rewards rejects non-numeric values):"
        )
        error_lines.append(json.dumps(stripped, indent=2, default=str))
    error_txt = reward_path.parent / "error.txt"
    error_txt.write_text("\n".join(error_lines) + "\n", encoding="utf-8")


def _write_zero_reward(
    reward_names: list[str],
    reward_path: Path,
    error: str,
) -> None:
    """Write a reward.json with all-zero scores; structured error to error.txt."""
    scores = {name: 0.0 for name in reward_names}
    scores["reward"] = 0.0
    _write_reward_artifacts(scores, reward_path, error=error)


def _run_verifier_mode(
    *, data_dir: Path, reward_path: Path, timeout: int
) -> None:
    """Run train.py and rely on harbor_submit to write reward.json.

    The held-out gt and the scoring logic both live on the host-side
    verifier process. The runner just needs to (a) run the agent, (b)
    confirm reward.json was written. Failure to find reward.json means
    the agent didn't call ``harbor_submit.submit_and_persist()`` — which
    is treated as an agent failure (zero reward + clear error).
    """
    train_path = _find_train_py()
    if train_path is None:
        # Generic-error zero reward: evaluator is host-side in this mode,
        # so we have no REWARD_NAMES to enumerate.
        _write_reward_artifacts(
            {"reward": 0.0},
            reward_path,
            error="No /workspace/train.py found",
        )
        print("[evaluate] No code found — reward = 0", file=sys.stderr)
        return

    print(
        f"[evaluate] verifier mode: VERIFIER_URL={os.environ.get('VERIFIER_URL')!r}"
    )
    print(f"[evaluate] Running agent code (timeout={timeout}s) ...")
    start = time.time()
    work_dir, success = _run_train(
        train_path,
        data_dir,
        timeout=timeout,
        extra_env={"HARBOR_REWARD_OUTPUT": str(reward_path)},
    )
    elapsed = time.time() - start
    print(f"[evaluate] Agent finished in {elapsed:.1f}s (success={success})")

    try:
        if not success:
            _write_reward_artifacts(
                {"reward": 0.0},
                reward_path,
                error="Agent code failed or timed out",
            )
            print("[evaluate] Agent failed — reward = 0", file=sys.stderr)
            return

        # Missing reward.json => agent contract failure (expected to call
        # harbor_submit.submit_and_persist()).
        if not reward_path.exists():
            _write_reward_artifacts(
                {"reward": 0.0},
                reward_path,
                error=(
                    "Agent did not write reward.json. In verifier mode "
                    "the agent must call "
                    "harbor_submit.submit_and_persist(predictions)."
                ),
            )
            print(
                "[evaluate] No reward.json produced — reward = 0",
                file=sys.stderr,
            )
            return

        # Normalise to single-key reward.json + metrics.json sidecar.
        # Agents POSTing to HARBOR_REWARD_OUTPUT directly may emit a legacy
        # multi-key dict; split it here to keep Mean.compute (len==1) happy.
        try:
            raw = json.loads(reward_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and len(raw) > 1:
                # Move the rich payload to metrics.json (filtered through
                # _numeric_only) and rewrite reward.json single-key.
                _write_reward_artifacts(raw, reward_path)
        except Exception as exc:  # noqa: BLE001 — diagnostic only
            print(
                f"[evaluate] failed to normalise reward.json: {exc}",
                file=sys.stderr,
            )
        print(
            f"[evaluate] reward.json present at {reward_path} "
            f"(elapsed={elapsed:.1f}s)"
        )
    finally:
        if work_dir is not None:
            shutil.rmtree(str(work_dir), ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenFinGym evaluation runner")
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument(
        "--eval-data-dir",
        default="/eval-data",
        help=(
            "Verifier-only mount holding the held-out test target "
            "(test_ground_truth.h5). Used only in the legacy file-IO "
            "fallback (when VERIFIER_URL is not set). New Phase 4 "
            "installs do not bake this directory into the agent image."
        ),
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--evaluator-path",
        required=True,
        help="Path to evaluator.py inside the task environment",
    )
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
    eval_data_dir = Path(args.eval_data_dir)
    reward_path = Path(args.reward_output)

    # VERIFIER_URL set => delegate scoring to the host verifier; only
    # run train.py with HARBOR_REWARD_OUTPUT and confirm reward.json.
    verifier_url = os.environ.get("VERIFIER_URL", "").strip()
    if verifier_url:
        return _run_verifier_mode(
            data_dir=data_dir,
            reward_path=reward_path,
            timeout=args.timeout,
        )

    # Legacy file-IO fallback: load the assembled evaluator and score locally.
    evaluator_path = Path(args.evaluator_path)
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
    if train_path is None:
        _write_zero_reward(
            evaluator.REWARD_NAMES, reward_path, "No /workspace/train.py found"
        )
        print("[evaluate] No code found — reward = 0")
        return

    # Run agent code
    print(f"[evaluate] Running agent code (timeout={args.timeout}s) ...")
    start = time.time()
    work_dir, success = _run_train(train_path, data_dir, timeout=args.timeout)
    elapsed = time.time() - start
    print(f"[evaluate] Agent finished in {elapsed:.1f}s (success={success})")

    try:
        if not success:
            _write_zero_reward(
                evaluator.REWARD_NAMES, reward_path, "Agent code failed or timed out"
            )
            print("[evaluate] Agent failed — reward = 0")
            return

        # Predictions only; the evaluator loads ground_truth itself from
        # eval_data_dir/test_ground_truth.h5 (agent has no access).
        predictions = _load_predictions(work_dir)
        ground_truth = None

        if predictions is None and _evaluator_requires_predictions(evaluator):
            print(
                "[evaluate] No common predictions file found; deferring to evaluator",
                file=sys.stderr,
            )

        # eval_data_dir is the verifier-only mount; the evaluator's
        # _load_reference_data reads test_ground_truth.h5 from there.
        # We point reward_output at metrics.json so the rich payload lands
        # there; reward.json (single-key, harbor-safe) is synthesised after.
        metrics_path = reward_path.parent / "metrics.json"
        reward_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            evaluator.score(
                predictions=predictions,
                ground_truth=ground_truth,
                weights=weights,
                reward_output=str(metrics_path),
                agent_output_dir=work_dir,
                data_dir=data_dir,
                eval_data_dir=eval_data_dir,
            )
        except Exception as exc:
            _write_zero_reward(
                evaluator.REWARD_NAMES,
                reward_path,
                f"Evaluator failed: {exc}",
            )
            print(f"[evaluate] Evaluator failed — reward = 0 ({exc})", file=sys.stderr)
            return
        # Synthesise the single-key reward.json from metrics.json: extract
        # the headline ``reward`` (falling back to weighted_total). Harbor's
        # Mean.compute requires len(dict)==1.
        try:
            if metrics_path.exists():
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    reward_value = payload.get("reward", payload.get("weighted_total", 0.0))
                    try:
                        reward_value = float(reward_value)
                    except (TypeError, ValueError):
                        reward_value = 0.0
                    reward_path.write_text(
                        json.dumps({"reward": reward_value}, indent=2),
                        encoding="utf-8",
                    )
        except Exception as exc:  # noqa: BLE001 — diagnostic only
            print(f"[evaluate] failed to write reward.json: {exc}", file=sys.stderr)
        print(f"[evaluate] Evaluation complete (elapsed={elapsed:.1f}s)")

    finally:
        if work_dir is not None:
            shutil.rmtree(str(work_dir), ignore_errors=True)


if __name__ == "__main__":
    main()
