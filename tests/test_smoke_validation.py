from __future__ import annotations

import json
from pathlib import Path

from openfinai_pipeline.benchmark.smoke_validation import (
    dry_fire_evaluator,
    validate_installed_task_bundle,
    write_dry_fire_failure_sidecar,
    write_validation_failure_sidecar,
)


def _write_installed_task(
    task_dir: Path,
    *,
    has_held_out_test_gt: bool = True,
    interaction_model: str = "forecasting",
) -> None:
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)
    data_dir = task_dir / "environment" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    eval_data_dir = task_dir / "environment" / "eval-data"
    eval_data_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "instruction.md").write_text("# Synthetic Task\n", encoding="utf-8")
    (task_dir / "task.toml").write_text('version = "1.0"\n', encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (task_dir / "environment" / "Dockerfile").write_text(
        # Faithful to the published image: no COPY data/ — data is
        # injected at runtime via Harbor's `mounts` mechanism.
        "FROM nihao0630/openfinai-base:v1\nWORKDIR /workspace\n",
        encoding="utf-8",
    )
    (data_dir / "task.py").write_text("VALUE = 1\n", encoding="utf-8")
    (data_dir / "task_rewards.py").write_text("# stub\n", encoding="utf-8")
    import h5py
    import numpy as np

    if has_held_out_test_gt:
        # B-shape forecasting / conditional generative install layout.
        # The trivial runtime loader is sufficient (no need to import
        # it from the runtime_loader module — the smoke gate only
        # consults ``load.py`` for unconditional generative tasks).
        (data_dir / "load.py").write_text(
            "def load(_):\n    return {}\n", encoding="utf-8"
        )
        with h5py.File(data_dir / "dataset.h5", "w") as _f:
            _f.attrs["split_policy"] = "chronological_80_20"
            train = _f.create_group("train")
            train.create_dataset(
                "features", data=np.zeros((4, 1), dtype=np.float64)
            )
            train.create_dataset(
                "ground_truth", data=np.zeros((4,), dtype=np.float64)
            )
            test = _f.create_group("test")
            test.create_dataset(
                "features", data=np.zeros((1, 1), dtype=np.float64)
            )
        with h5py.File(eval_data_dir / "test_ground_truth.h5", "w") as _f:
            gtg = _f.create_group("ground_truth")
            gtg.create_dataset(
                "_value", data=np.zeros((1,), dtype=np.float64)
            )
    else:
        # Unconditional generative — load.py returns reference; no
        # held-out artifact and the slicer would have written
        # dataset.h5 with a top-level 'reference' group.
        (data_dir / "load.py").write_text(
            "import numpy as np\n"
            "def load(_):\n"
            "    return {\n"
            '        "reference": np.zeros((3,), dtype=np.float64),\n'
            '        "split_policy": "no_split",\n'
            '        "split_metadata": None,\n'
            "    }\n",
            encoding="utf-8",
        )
        with h5py.File(data_dir / "dataset.h5", "w") as _f:
            _f.attrs["split_policy"] = "no_split"
            _f.create_dataset(
                "reference", data=np.zeros((3,), dtype=np.float64)
            )
    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "task_id": "task1",
                "scope_id": "scope1",
                "interaction_model": interaction_model,
                "class_name": "Task1",
                "evaluator_class_name": "SyntheticEvaluator",
                "version": "0.1.0",
                "resolved_reward_names": [],
                "expected_score_keys": ["reward"],
                "expected_agent_output_filenames": [
                    "predictions.npy",
                    "predictions.csv",
                ],
                "has_held_out_test_gt": has_held_out_test_gt,
                "split_policy": (
                    "chronological_80_20" if has_held_out_test_gt else "no_split"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "__init__.py").write_text("", encoding="utf-8")
    (data_dir / "evaluator.py").write_text(
        """
import json
from pathlib import Path


class SyntheticEvaluator:
    REWARD_NAMES = ["reward"]

    def score(self, predictions, ground_truth, weights=None, reward_output=None, **kwargs):
        score = {"reward": 1.0}
        if reward_output is not None:
            out = Path(reward_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(score), encoding="utf-8")
        return score
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_smoke_gate_passes_on_synthetic_bundle(tmp_path):
    task_dir = tmp_path / "benchmark" / "scope1" / "task1"
    _write_installed_task(task_dir)

    result = validate_installed_task_bundle(
        task_dir, scope_id="scope1", task_id="task1"
    )

    assert result["valid"] is True, result
    assert result["errors"] == []
    assert result["task"] == "scope1__task1"
    assert result["scope_id"] == "scope1"
    assert result["task_id"] == "task1"
    assert result["layout_checked"]
    assert result["smoke"]["ran"] is True
    assert result["smoke"]["score_succeeded"] is True
    assert result["smoke"]["evaluator_path"].endswith("environment/data/evaluator.py")
    assert result["smoke"]["data_dir"].endswith("environment/data")
    assert result["smoke"]["eval_data_dir"].endswith("environment/eval-data")
    reward_path = Path(result["smoke"]["reward_path"])
    assert reward_path.exists()
    reward_payload = json.loads(reward_path.read_text(encoding="utf-8"))
    assert reward_payload["reward"] == 1.0
    assert result["smoke"]["scores"]["reward"] == 1.0


def test_smoke_gate_passes_for_unconditional_generative(tmp_path):
    """``has_held_out_test_gt=False`` reads the reference via ``load.py``."""
    task_dir = tmp_path / "benchmark" / "scope1" / "task_uncond"
    _write_installed_task(
        task_dir,
        has_held_out_test_gt=False,
        interaction_model="generative",
    )

    result = validate_installed_task_bundle(
        task_dir, scope_id="scope1", task_id="task_uncond"
    )

    assert result["valid"] is True, result
    assert result["smoke"]["score_succeeded"] is True
    assert result["smoke"]["scores"]["reward"] == 1.0


def test_smoke_gate_reports_missing_required_files(tmp_path):
    task_dir = tmp_path / "broken_task"
    task_dir.mkdir(parents=True, exist_ok=True)

    result = validate_installed_task_bundle(
        task_dir, scope_id="scope1", task_id="broken"
    )

    assert result["valid"] is False
    assert any(str(error).startswith("missing:instruction.md") for error in result["errors"])
    # Smoke must NOT run when the layout pre-check failed.
    assert result["smoke"]["ran"] is False


def test_write_validation_failure_sidecar_uses_posix_paths(tmp_path):
    tasks_root = tmp_path / "benchmark"
    task_dir = tasks_root / "scope1" / "task1"
    task_dir.mkdir(parents=True, exist_ok=True)

    sidecar = write_validation_failure_sidecar(
        task_dir,
        tasks_root=tasks_root,
        scope_id="scope1",
        task_id="task1",
        result={"errors": ["smoke_train_failed"], "smoke": {"ran": True}},
    )

    assert sidecar == task_dir / "validation_failed.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["phase_tag"] == "phase4:benchmark"
    assert payload["scope_id"] == "scope1"
    assert payload["task_id"] == "task1"
    # POSIX path with forward slashes, no backslashes even on Windows.
    assert payload["task_dir"] == "scope1/task1"
    assert "\\" not in payload["task_dir"]
    assert payload["errors"] == ["smoke_train_failed"]
    assert payload["smoke"] == {"ran": True}
    assert payload["validated_at_utc"].endswith("Z")


def test_write_validation_failure_sidecar_handles_unrelated_root(tmp_path):
    """Falls back to absolute POSIX path when relative_to() fails."""
    task_dir = tmp_path / "real_root" / "scope1" / "task1"
    unrelated_root = tmp_path / "other_root"
    task_dir.mkdir(parents=True, exist_ok=True)
    unrelated_root.mkdir(parents=True, exist_ok=True)

    sidecar = write_validation_failure_sidecar(
        task_dir,
        tasks_root=unrelated_root,
        scope_id="scope1",
        task_id="task1",
        result={"errors": [], "smoke": {}},
    )

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    # Fell back to absolute path; still POSIX-encoded.
    assert "\\" not in payload["task_dir"]
    assert payload["task_dir"].endswith("scope1/task1")


# Dry-fire gate (pre-implementation evaluator dry-run on staging)

_SYNTHETIC_LOAD_PY = """\
import numpy as np
from pathlib import Path


def load(data_dir):
    data_dir = Path(data_dir)
    raw = np.loadtxt(
        data_dir / "series.csv", delimiter=",", skiprows=1
    ).astype(np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    features = raw.astype(np.float64)
    # Synthetic shifted-target ground truth: deterministic, finite, aligned.
    gt = np.roll(features[:, 0], -1).astype(np.float64)
    gt[-1] = features[-1, 0]
    cut = max(1, int(0.8 * len(features)))
    return {
        "train": {"features": features[:cut], "ground_truth": gt[:cut]},
        "test": {"features": features[cut:], "ground_truth": gt[cut:]},
        "split_policy": "chronological_80_20",
        "split_metadata": None,
    }
"""

_SYNTHETIC_EVALUATOR_PASSING = """\
import json
from pathlib import Path
import numpy as np

try:
    from openfinai_pipeline.benchmark.contracts import BaseEvaluator
except Exception:
    class BaseEvaluator:
        REWARD_NAMES = []


class SyntheticEvaluator(BaseEvaluator):
    REWARD_NAMES = ["mse_loss"]

    def score(self, predictions, ground_truth, weights=None, reward_output=None, **kwargs):
        gt = np.asarray(ground_truth, dtype=np.float64)
        preds = np.asarray(predictions, dtype=np.float64)
        mse = float(np.mean((preds - gt) ** 2))
        scores = {"mse_loss": mse, "reward": mse}
        if reward_output is not None:
            out = Path(reward_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(scores), encoding="utf-8")
        return scores
"""

_SYNTHETIC_EVALUATOR_RAISING = """\
import numpy as np

try:
    from openfinai_pipeline.benchmark.contracts import BaseEvaluator
except Exception:
    class BaseEvaluator:
        REWARD_NAMES = []


class SyntheticEvaluator(BaseEvaluator):
    REWARD_NAMES = ["mse_loss"]

    def score(self, predictions, ground_truth, weights=None, reward_output=None, **kwargs):
        # Simulate a reward that broadcasts on incompatible shapes — i.e. the
        # exact failure mode harbor smoke catches today (reward x loader-output
        # shape mismatch). The dry-fire must surface this as a raise, not
        # silently complete.
        np.asarray(predictions).reshape(-1, 99) - np.asarray(ground_truth).reshape(-1, 1)
        return {"mse_loss": 0.0, "reward": 0.0}
"""

_SYNTHETIC_EVALUATOR_BLIND_ONLY = """\
import json
from pathlib import Path
import numpy as np

try:
    from openfinai_pipeline.benchmark.contracts import BaseEvaluator
except Exception:
    class BaseEvaluator:
        REWARD_NAMES = []


class SyntheticEvaluator(BaseEvaluator):
    # Reproduces the chinaa_symbolicalphadiscovery failure mode: the
    # single configured metric "informationcorrelation" reshapes (N,)
    # predictions to (N, 1) and centers across the singleton cross-
    # section dim, so its output is exactly 0.0 for ANY 1-D input.
    REWARD_NAMES = ["informationcorrelation"]

    def score(self, predictions, ground_truth, weights=None, reward_output=None, **kwargs):
        pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
        gt = np.asarray(ground_truth, dtype=np.float64).reshape(-1)
        n = pred.shape[0]
        if n == 0:
            return {"informationcorrelation": 0.0, "reward": 0.0}
        pred_panel = pred.reshape(n, -1)
        gt_panel = gt.reshape(n, -1)
        pred_centered = pred_panel - pred_panel.mean(axis=1, keepdims=True)
        gt_centered = gt_panel - gt_panel.mean(axis=1, keepdims=True)
        cov = (pred_centered * gt_centered).mean(axis=1)
        ic = float(cov.mean())
        scores = {"informationcorrelation": ic, "reward": ic}
        if reward_output is not None:
            out = Path(reward_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(scores), encoding="utf-8")
        return scores
"""

_SYNTHETIC_EVALUATOR_BLIND_PLUS_SENSITIVE = """\
import json
from pathlib import Path
import numpy as np

try:
    from openfinai_pipeline.benchmark.contracts import BaseEvaluator
except Exception:
    class BaseEvaluator:
        REWARD_NAMES = []


class SyntheticEvaluator(BaseEvaluator):
    # Mixed bundle: one prediction-sensitive metric, one blind.
    # Reproduces the tasks 1 & 2 silent-pass case where the weighted
    # Aggregate totals differ even though one metric is prediction-blind.
    # while a configured metric is in fact prediction-blind. The new
    # per-metric probe must catch the blind one and leave mse_loss.
    REWARD_NAMES = ["mse_loss", "informationcorrelation"]

    def score(self, predictions, ground_truth, weights=None, reward_output=None, **kwargs):
        pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
        gt = np.asarray(ground_truth, dtype=np.float64).reshape(-1)
        mse = float(np.mean((pred - gt) ** 2)) if pred.size else 0.0
        n = pred.shape[0]
        if n == 0:
            ic = 0.0
        else:
            pred_panel = pred.reshape(n, -1)
            gt_panel = gt.reshape(n, -1)
            pred_centered = pred_panel - pred_panel.mean(axis=1, keepdims=True)
            gt_centered = gt_panel - gt_panel.mean(axis=1, keepdims=True)
            cov = (pred_centered * gt_centered).mean(axis=1)
            ic = float(cov.mean())
        scores = {"mse_loss": mse, "informationcorrelation": ic, "reward": mse + ic}
        if reward_output is not None:
            out = Path(reward_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(scores), encoding="utf-8")
        return scores
"""

_SYNTHETIC_EVALUATOR_PER_REWARD_ERROR = """\
import json
import math
from pathlib import Path

try:
    from openfinai_pipeline.benchmark.contracts import BaseEvaluator
except Exception:
    class BaseEvaluator:
        REWARD_NAMES = []


class SyntheticEvaluator(BaseEvaluator):
    REWARD_NAMES = ["mse_loss"]

    def score(self, predictions, ground_truth, weights=None, reward_output=None, **kwargs):
        # Simulate the assembled-template's silent-failure path: per-reward
        # error captured into _errors, NaN scalar in scores. The dry-fire
        # must reject this even though the call returned a dict.
        scores = {
            "mse_loss": float("nan"),
            "reward": 0.0,
            "_errors": {"mse_loss": "ValueError: synthetic broken metric"},
        }
        if reward_output is not None:
            out = Path(reward_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(scores, default=str), encoding="utf-8")
        return scores
"""


def _write_synthetic_staging(
    staging_dir: Path,
    *,
    evaluator_source: str = _SYNTHETIC_EVALUATOR_PASSING,
    include_load: bool = True,
) -> Path:
    """Materialise a minimal staging dir + dataset data dir.

    Returns the dataset's ``data/`` directory so callers can pass it
    as ``data_dir`` to :func:`dry_fire_evaluator`.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "evaluator.py").write_text(evaluator_source, encoding="utf-8")
    if include_load:
        (staging_dir / "load.py").write_text(_SYNTHETIC_LOAD_PY, encoding="utf-8")

    data_dir = staging_dir.parent / "_dataset" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "series.csv").write_text(
        "value\n1.0\n2.0\n3.0\n4.0\n5.0\n", encoding="utf-8"
    )
    return data_dir


def test_dry_fire_passes_on_clean_staging(tmp_path):
    staging_dir = tmp_path / "staging" / "task_synth"
    data_dir = _write_synthetic_staging(staging_dir)

    result = dry_fire_evaluator(
        staging_dir=staging_dir,
        data_dir=data_dir,
        resolved_reward_names=["mse_loss"],
    )

    assert result["ran"] is True, result
    assert result["ok"] is True, result["errors"]
    assert result["errors"] == []
    assert "mse_loss" in result["score_keys"]
    assert "reward" in result["score_keys"]
    # Reward output landed in the dry-fire dir and is parseable JSON.
    reward_path = Path(result["reward_path"])
    assert reward_path.exists()
    payload = json.loads(reward_path.read_text(encoding="utf-8"))
    assert payload["mse_loss"] == result["scores"]["mse_loss"]
    # Dry-fire dir was materialised under the staging root.
    assert result["dry_fire_dir"].endswith("/.dry_fire") or result[
        "dry_fire_dir"
    ].endswith("\\.dry_fire")


def test_dry_fire_fails_when_load_py_missing(tmp_path):
    staging_dir = tmp_path / "staging" / "task_no_loader"
    data_dir = _write_synthetic_staging(staging_dir, include_load=False)

    result = dry_fire_evaluator(
        staging_dir=staging_dir,
        data_dir=data_dir,
        resolved_reward_names=["mse_loss"],
    )

    assert result["ran"] is False
    assert result["ok"] is False
    assert any(
        err.startswith("missing:staging/load.py") for err in result["errors"]
    ), result["errors"]
    # No score output should have been written.
    assert result["scores"] == {}


def test_dry_fire_fails_when_evaluator_missing(tmp_path):
    staging_dir = tmp_path / "staging" / "task_no_eval"
    staging_dir.mkdir(parents=True, exist_ok=True)
    # Loader present but evaluator absent — gate must reject before
    # touching the loader.
    (staging_dir / "load.py").write_text(_SYNTHETIC_LOAD_PY, encoding="utf-8")
    data_dir = staging_dir.parent / "_dataset" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "series.csv").write_text("v\n1.0\n", encoding="utf-8")

    result = dry_fire_evaluator(
        staging_dir=staging_dir,
        data_dir=data_dir,
        resolved_reward_names=["mse_loss"],
    )

    assert result["ran"] is False
    assert result["ok"] is False
    assert any(
        err.startswith("missing:staging/evaluator.py") for err in result["errors"]
    ), result["errors"]


def test_dry_fire_catches_score_raise(tmp_path):
    """Evaluator that crashes inside score() — i.e. shape mismatch — fails the gate."""
    staging_dir = tmp_path / "staging" / "task_raises"
    data_dir = _write_synthetic_staging(
        staging_dir, evaluator_source=_SYNTHETIC_EVALUATOR_RAISING
    )

    result = dry_fire_evaluator(
        staging_dir=staging_dir,
        data_dir=data_dir,
        resolved_reward_names=["mse_loss"],
    )

    assert result["ran"] is False, result
    assert result["ok"] is False
    assert any(err.startswith("score_raised:") for err in result["errors"]), result[
        "errors"
    ]


def test_dry_fire_catches_per_reward_error(tmp_path):
    """Evaluator returns NaN+_errors instead of raising — gate still rejects."""
    staging_dir = tmp_path / "staging" / "task_per_reward_err"
    data_dir = _write_synthetic_staging(
        staging_dir, evaluator_source=_SYNTHETIC_EVALUATOR_PER_REWARD_ERROR
    )

    result = dry_fire_evaluator(
        staging_dir=staging_dir,
        data_dir=data_dir,
        resolved_reward_names=["mse_loss"],
    )

    assert result["ran"] is True
    assert result["ok"] is False
    # Two failure modes both surface: the per-reward error AND the NaN value.
    assert any(
        err.startswith("reward_error:mse_loss:") for err in result["errors"]
    ), result["errors"]
    assert any(
        err.startswith("reward_value_non_finite:mse_loss") for err in result["errors"]
    ), result["errors"]


def test_dry_fire_flags_prediction_blind_metric(tmp_path):
    """Reproduces the chinaa_symbolicalphadiscovery failure mode.

    A metric that returns the same finite value for ``score(zeros, gt)``
    and ``score(gt, gt)`` is prediction-blind on this loader's output
    shape. The dry-fire's per-metric probe must flag it via
    ``prediction_blind`` and add a ``prediction_blind_metrics:`` failure
    marker — even though every other dispatch / value check passes.
    """
    staging_dir = tmp_path / "staging" / "task_blind_only"
    data_dir = _write_synthetic_staging(
        staging_dir, evaluator_source=_SYNTHETIC_EVALUATOR_BLIND_ONLY
    )

    result = dry_fire_evaluator(
        staging_dir=staging_dir,
        data_dir=data_dir,
        resolved_reward_names=["informationcorrelation"],
    )

    assert result["ran"] is True, result
    assert result["ok"] is False, result
    assert result["prediction_blind"] == ["informationcorrelation"], result
    assert any(
        err.startswith("prediction_blind_metrics:") for err in result["errors"]
    ), result["errors"]
    # The zeros-side dispatch / finite checks must NOT trigger — the
    # blindness probe is the ONLY reason this gate fails. Otherwise
    # callers can't distinguish "prune-and-retry" from a hard failure.
    assert not any(
        err.startswith("reward_dispatch_missing:") for err in result["errors"]
    ), result["errors"]
    assert not any(
        err.startswith("reward_value_non_finite:") for err in result["errors"]
    ), result["errors"]


def test_dry_fire_flags_only_blind_metric_in_mixed_bundle(tmp_path):
    """The dry fire isolates a blind metric from a sensitive one."""
    staging_dir = tmp_path / "staging" / "task_blind_plus_sensitive"
    data_dir = _write_synthetic_staging(
        staging_dir, evaluator_source=_SYNTHETIC_EVALUATOR_BLIND_PLUS_SENSITIVE
    )

    result = dry_fire_evaluator(
        staging_dir=staging_dir,
        data_dir=data_dir,
        resolved_reward_names=["mse_loss", "informationcorrelation"],
    )

    assert result["ran"] is True, result
    assert result["ok"] is False, result
    assert result["prediction_blind"] == ["informationcorrelation"], result
    # mse_loss is sensitive: zeros vs gt produce different finite scores.
    assert "mse_loss" not in result["prediction_blind"]
    # The probe-driven failure marker mentions only the blind metric.
    blind_errors = [
        err for err in result["errors"] if err.startswith("prediction_blind_metrics:")
    ]
    assert len(blind_errors) == 1, result["errors"]
    assert "informationcorrelation" in blind_errors[0]
    assert "mse_loss" not in blind_errors[0]


def test_dry_fire_catches_resolved_reward_missing_from_output(tmp_path):
    """Assembler claimed a reward that the evaluator does not actually score."""
    staging_dir = tmp_path / "staging" / "task_missing_reward"
    data_dir = _write_synthetic_staging(staging_dir)

    # Evaluator scores 'mse_loss' but assembler claimed 'mse_loss' AND
    # 'directional_accuracy'. The latter must surface as a dispatch miss.
    result = dry_fire_evaluator(
        staging_dir=staging_dir,
        data_dir=data_dir,
        resolved_reward_names=["mse_loss", "directional_accuracy"],
    )

    assert result["ran"] is True
    assert result["ok"] is False
    assert any(
        err == "reward_dispatch_missing:directional_accuracy"
        for err in result["errors"]
    ), result["errors"]
    # mse_loss was actually wired up — that one shouldn't be flagged.
    assert not any(
        err == "reward_dispatch_missing:mse_loss" for err in result["errors"]
    )


def test_dry_fire_rebuilds_dry_fire_dir_each_run(tmp_path):
    """Stale state from a prior dry-fire must not bleed into a new attempt."""
    staging_dir = tmp_path / "staging" / "task_rerun"
    data_dir = _write_synthetic_staging(staging_dir)

    first = dry_fire_evaluator(
        staging_dir=staging_dir,
        data_dir=data_dir,
        resolved_reward_names=["mse_loss"],
    )
    assert first["ok"] is True
    dry_fire_root = Path(first["dry_fire_dir"])
    # Drop a sentinel file inside .dry_fire/ — the next run should wipe it.
    sentinel = dry_fire_root / "__stale_sentinel__"
    sentinel.write_text("stale", encoding="utf-8")
    assert sentinel.exists()

    second = dry_fire_evaluator(
        staging_dir=staging_dir,
        data_dir=data_dir,
        resolved_reward_names=["mse_loss"],
    )
    assert second["ok"] is True
    assert not sentinel.exists(), (
        "dry_fire_evaluator did not wipe stale state from a prior run"
    )


def test_write_dry_fire_failure_sidecar_format(tmp_path):
    staging_dir = tmp_path / "staging" / "scope1__task1"
    staging_dir.mkdir(parents=True, exist_ok=True)
    fake_result = {
        "ran": True,
        "ok": False,
        "errors": ["reward_dispatch_missing:directional_accuracy"],
        "evaluator_path": (staging_dir / "evaluator.py").as_posix(),
        "load_path": (staging_dir / "load.py").as_posix(),
        "data_dir": (tmp_path / "_dataset" / "data").as_posix(),
        "dry_fire_dir": (staging_dir / ".dry_fire").as_posix(),
        "reward_path": (staging_dir / ".dry_fire" / "_reward.json").as_posix(),
        "scores": {"mse_loss": 0.5, "reward": 0.5},
        "score_keys": ["mse_loss", "reward"],
    }

    sidecar = write_dry_fire_failure_sidecar(
        staging_dir,
        scope_id="scope1",
        task_id="task1",
        result=fake_result,
    )

    assert sidecar == staging_dir / "dry_fire_failed.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["phase_tag"] == "phase4:benchmark"
    assert payload["gate"] == "dry_fire"
    assert payload["scope_id"] == "scope1"
    assert payload["task_id"] == "task1"
    assert payload["errors"] == ["reward_dispatch_missing:directional_accuracy"]
    assert payload["ran"] is True
    assert payload["score_keys"] == ["mse_loss", "reward"]
    assert payload["validated_at_utc"].endswith("Z")
    # Latest-wins overwrite.
    sidecar2 = write_dry_fire_failure_sidecar(
        staging_dir,
        scope_id="scope1",
        task_id="task1",
        result={**fake_result, "errors": ["different_error"]},
    )
    assert sidecar2 == sidecar
    payload2 = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload2["errors"] == ["different_error"]
