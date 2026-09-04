"""Shared helpers for the static Phase 4 test files.

This module is staged into ``staging_dir`` next to ``test_task.py``. It
loads the per-task ``manifest.json`` and provides deterministic in-memory
fixtures so tests can exercise the task without touching the real
``load.py`` (the bundled loader is exercised separately by the post-install
smoke gate).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def load_manifest() -> Dict[str, Any]:
    """Read ``manifest.json`` from this file's directory.

    The manifest is a per-task descriptor written deterministically by
    Phase 4 staging. It carries the canonical class names, expected
    score-dict keys, and expected agent-output filenames -- everything
    that varies per task so a single static test file can drive every
    task of a given interaction model.
    """
    manifest_path = Path(__file__).resolve().parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json missing at {manifest_path}; was the staging "
            "step _write_task_manifest invoked?"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def import_task_class(manifest: Dict[str, Any]) -> type:
    """Return the task subclass via static import + getattr.

    Uses a regular ``import task`` (static AST node) so we do NOT touch
    ``importlib`` -- which is in the sandbox's blocked-modules set. The
    class name varies per task (PascalCased ``task_id``), so the lookup
    is dynamic via ``getattr``.
    """
    import task as _task_module  # noqa: WPS433 — staged in sys.path[cwd=staging]
    class_name = manifest["class_name"]
    cls = getattr(_task_module, class_name, None)
    if cls is None:
        raise AttributeError(
            f"task module does not export {class_name!r}; available: "
            f"{[n for n in dir(_task_module) if not n.startswith('_')]}"
        )
    return cls


def import_evaluator_class(manifest: Dict[str, Any]) -> type:
    """Return the evaluator class. Same pattern as ``import_task_class``."""
    import evaluator as _eval_module  # noqa: WPS433
    class_name = manifest["evaluator_class_name"]
    cls = getattr(_eval_module, class_name, None)
    if cls is None:
        raise AttributeError(
            f"evaluator module does not export {class_name!r}; available: "
            f"{[n for n in dir(_eval_module) if not n.startswith('_')]}"
        )
    return cls


def build_forecasting_fixture(
    n_samples_train: int = 16,
    n_samples_test: int = 8,
    n_features: int = 3,
    seed: int = 0,
) -> Dict[str, Any]:
    """Build a synthetic B-shape forecasting fixture.

    Returns the same nested ``{"train": {features, ground_truth},
    "test": {features, ground_truth}}`` shape the install slicer
    materialises into ``dataset.h5``. Ground-truths are deterministic
    non-zero random draws so the F5 anti-degenerate test can
    distinguish zero-prediction from perfect-prediction.

    Test-side ``ground_truth`` is included in this fixture (unlike
    the on-disk shape, which omits it) so the static tests can verify
    end-to-end scoring against the held-out target without needing
    a separate ``test_ground_truth.h5`` artifact.
    """
    rng = np.random.default_rng(seed)
    return {
        "train": {
            "features": rng.standard_normal((n_samples_train, n_features)).astype(np.float32),
            "ground_truth": rng.standard_normal(n_samples_train).astype(np.float32),
        },
        "test": {
            "features": rng.standard_normal((n_samples_test, n_features)).astype(np.float32),
            "ground_truth": rng.standard_normal(n_samples_test).astype(np.float32),
        },
        "split_policy": "chronological_80_20",
        "split_metadata": None,
    }


def build_generative_fixture(
    n_samples_train: int = 16,
    n_samples_test: int = 8,
    n_features: int = 3,
    n_conditioning: int = 2,
    seed: int = 0,
) -> Dict[str, Any]:
    """Build a synthetic B-shape generative fixture (conditional case).

    Mirrors the conditional generative task data layout. ``features``
    is the conditioning input; ``ground_truth`` is the real reference
    samples. Test-side ``ground_truth`` is present here for the test
    to score against, but on disk it would be held out.
    """
    rng = np.random.default_rng(seed)
    return {
        "train": {
            "features": rng.standard_normal((n_samples_train, n_conditioning)).astype(np.float32),
            "ground_truth": rng.standard_normal((n_samples_train, n_features)).astype(np.float32),
        },
        "test": {
            "features": rng.standard_normal((n_samples_test, n_conditioning)).astype(np.float32),
            "ground_truth": rng.standard_normal((n_samples_test, n_features)).astype(np.float32),
        },
        "split_policy": "chronological_80_20",
        "split_metadata": None,
    }


def make_forecasting_task(manifest: Dict[str, Any], fixture: Dict[str, Any]):
    """Instantiate the task and inject the fixture (no load.py involvement)."""
    cls = import_task_class(manifest)
    task = cls()
    task._data = fixture
    return task


def make_generative_task(manifest: Dict[str, Any], fixture: Dict[str, Any]):
    """Instantiate the task and inject the fixture (no load.py involvement)."""
    cls = import_task_class(manifest)
    task = cls()
    task._data = fixture
    return task


def wire_evaluator(task, manifest: Dict[str, Any]) -> Any:
    """Attach a freshly built evaluator instance to the task.

    ``ForecastingTask.predict_and_evaluate`` and
    ``GenerativeTask.generate_and_evaluate`` only dispatch to the
    assembled evaluator when ``task._evaluator`` is set; otherwise they
    fall back to a default scorer. The static tests want to exercise the
    real assembled evaluator, so we wire it explicitly.
    """
    evaluator_cls = import_evaluator_class(manifest)
    evaluator = evaluator_cls()
    task._evaluator = evaluator
    return evaluator
