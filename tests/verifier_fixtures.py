"""Build self-contained synthetic task layouts for verifier tests.

Layout produced::

    <tmp>/tasks/scope/synthetic_task/
      manifest.json (top-level convenience copy)
      environment/
        data/
          evaluator.py       (synthetic evaluator, mirrors the contract)
          manifest.json
        eval-data/
          gt.npy             (held-out test ground truth)

The evaluator loads gt.npy, returns mse and reward, and records forwarded kwargs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SYNTHETIC_EVALUATOR_SOURCE = '''\
"""Synthetic evaluator for the verifier test suite.

Mirrors the assembled-evaluator contract: REWARD_NAMES list, score(...)
method, optional module-level _load_reference_data helper that the
verifier's _preload_ground_truth() picks up. Whether to actually call
the loader is decided by the server from cfg.has_held_out_test_gt
(read from manifest.json), so no _HAS_GROUND_TRUTH attribute is
needed here.
"""
from pathlib import Path

import numpy as np


_REWARD_SPECS = []  # tested separately; not exercised by these fixtures


def _load_reference_data(eval_data_dir, **_):
    p = Path(eval_data_dir) / "gt.npy"
    if not p.exists():
        return None
    return np.load(str(p))


class SyntheticEvaluator:
    """Deterministic MSE-style scorer used by verifier tests."""

    REWARD_NAMES = ["mse"]
    EXPECTS_PREDICTIONS = True
    EXPECTS_GROUND_TRUTH = True

    def score(
        self,
        predictions=None,
        ground_truth=None,
        weights=None,
        reward_output=None,
        **kwargs,
    ):
        if predictions is None or ground_truth is None:
            return {
                "mse": 0.0,
                "reward": 0.0,
                "_kwargs_seen": sorted(kwargs.keys()),
            }
        pred = np.asarray(predictions, dtype=float).ravel()
        gt = np.asarray(ground_truth, dtype=float).ravel()
        n = min(pred.shape[0], gt.shape[0])
        mse = float(np.mean((pred[:n] - gt[:n]) ** 2))
        return {
            "mse": mse,
            "reward": -mse,
            "_kwargs_seen": sorted(kwargs.keys()),
        }
'''


# A second flavour: evaluator that raises on score() to test 500 path.
SYNTHETIC_EVALUATOR_RAISES_SOURCE = '''\
"""Synthetic evaluator that raises on score() — for error-path tests."""
from pathlib import Path

import numpy as np


def _load_reference_data(eval_data_dir, **_):
    p = Path(eval_data_dir) / "gt.npy"
    if not p.exists():
        return None
    return np.load(str(p))


class SyntheticEvaluatorRaises:
    REWARD_NAMES = ["mse"]
    EXPECTS_PREDICTIONS = True
    EXPECTS_GROUND_TRUTH = True

    def score(self, predictions=None, ground_truth=None, **kwargs):
        raise ValueError("synthetic explosion in score()")
'''


def make_synthetic_task(
    tmp_path: Path,
    *,
    task_id: str = "synthetic_task",
    scope: str = "scope",
    evaluator_source: str = SYNTHETIC_EVALUATOR_SOURCE,
    gt_array: np.ndarray | None = None,
    interaction_model: str = "forecasting",
    has_held_out_test_gt: bool = True,
) -> Path:
    """Build a self-contained synthetic task layout. Returns ``task_dir``."""
    if gt_array is None:
        gt_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    task_dir = tmp_path / "tasks" / scope / task_id
    data_dir = task_dir / "environment" / "data"
    eval_dir = task_dir / "environment" / "eval-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    (data_dir / "evaluator.py").write_text(evaluator_source, encoding="utf-8")
    (data_dir / "__init__.py").write_text("", encoding="utf-8")

    np.save(str(eval_dir / "gt.npy"), gt_array)

    manifest = {
        "schema_version": 2,
        "task_id": task_id,
        "scope_id": scope,
        "interaction_model": interaction_model,
        "class_name": "Synthetic",
        "evaluator_class_name": "SyntheticEvaluator",
        "version": "0.1.0",
        "resolved_reward_names": ["mse"],
        "expected_score_keys": ["mse", "reward"],
        "expected_agent_output_filenames": ["predictions.npy", "predictions.csv"],
        "has_held_out_test_gt": has_held_out_test_gt,
        "split_policy": "chronological_80_20",
    }
    manifest_str = json.dumps(manifest, indent=2)
    (data_dir / "manifest.json").write_text(manifest_str, encoding="utf-8")

    return task_dir


def encode_array_b64(arr: np.ndarray) -> str:
    """Convenience: base64-encode a numpy array's .npy bytes."""
    import base64
    import io

    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")
