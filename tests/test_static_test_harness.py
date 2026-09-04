"""Contracts for deterministic manifests, staged tests, and schemas.

The end-to-end case runs the checked-in static tests against a synthetic task.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from openfinai_pipeline.benchmark.builders.task_builder import (
    GeneratedCode,
    TaskCandidate,
    _canonical_class_name,
    _stage_static_test_files,
    _write_task_manifest,
)
from openfinai_pipeline.prompts.task_builder import (
    task_fix_schema,
    task_generation_schema,
)


# Fixtures


def _make_candidate(task_id: str = "demo_task", scope_id: str = "demo_scope") -> TaskCandidate:
    return TaskCandidate(
        task_dir="",
        task_id=task_id,
        title="Demo Task",
        scope_id=scope_id,
        paper_id="paper_x",
        ml_task_summary="A short summary",
        experiments="",
        datasets=[],
        rewards=[],
        task_family="forecasting",
        paper_abstract="",
        pdf_path=None,
    )


# Schema and dataclass contracts


class TestSchemaDropsTestPy:
    def test_generated_code_has_no_test_py_field(self):
        names = {f.name for f in fields(GeneratedCode)}
        assert "test_py" not in names, (
            "GeneratedCode still has a test_py field; it must be removed "
            "to prevent LLM-authored test files from being staged."
        )

    def test_task_generation_schema_drops_test_py(self):
        schema = task_generation_schema()
        assert "test_py" not in schema["properties"]
        assert "test_py" not in schema["required"]

    def test_task_fix_schema_drops_test_py(self):
        schema = task_fix_schema()
        assert "test_py" not in schema["properties"]


# Canonical class name


class TestCanonicalClassName:
    @pytest.mark.parametrize(
        ("task_id", "expected"),
        [
            ("task_1", "Task1"),
            ("task-1", "Task-1"),  # only underscore is normalized; hyphen survives
            ("acl18", "Acl18"),
            ("offline_crypto_forecasting", "OfflineCryptoForecasting"),
            ("xgboost_btc_v2", "XgboostBtcV2"),
        ],
    )
    def test_known_inputs(self, task_id, expected):
        assert _canonical_class_name(task_id) == expected


# _write_task_manifest


class TestWriteTaskManifestForecasting:
    def test_writes_expected_schema_and_values(self, tmp_path):
        candidate = _make_candidate(task_id="demo_task", scope_id="forecast_scope")
        out = _write_task_manifest(
            tmp_path,
            candidate,
            "forecasting",
            ["MSELoss", "DirectionalAccuracy"],
        )
        assert out == tmp_path / "manifest.json"
        payload = json.loads(out.read_text(encoding="utf-8"))

        assert payload["schema_version"] == 2
        assert payload["task_id"] == "demo_task"
        assert payload["scope_id"] == "forecast_scope"
        assert payload["interaction_model"] == "forecasting"
        assert payload["class_name"] == "DemoTask"
        assert payload["evaluator_class_name"] == "DemoTaskEvaluator"
        assert payload["version"] == "0.1.0"
        assert payload["resolved_reward_names"] == ["MSELoss", "DirectionalAccuracy"]
        assert payload["expected_score_keys"] == [
            "directionalaccuracy",
            "mseloss",
            "reward",
        ]
        assert payload["expected_agent_output_filenames"] == [
            "predictions.npy",
            "predictions.csv",
        ]

    def test_empty_reward_list_still_includes_reward_sentinel(self, tmp_path):
        candidate = _make_candidate()
        out = _write_task_manifest(tmp_path, candidate, "forecasting", [])
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["expected_score_keys"] == ["reward"]


class TestWriteTaskManifestGenerative:
    def test_writes_expected_schema_and_values(self, tmp_path):
        candidate = _make_candidate(task_id="market_gen", scope_id="gen_scope")
        out = _write_task_manifest(
            tmp_path,
            candidate,
            "generative",
            ["VARMetricLoss"],
        )
        payload = json.loads(out.read_text(encoding="utf-8"))

        assert payload["interaction_model"] == "generative"
        assert payload["class_name"] == "MarketGen"
        assert payload["evaluator_class_name"] == "MarketGenEvaluator"
        assert payload["expected_agent_output_filenames"] == [
            "samples.npy",
            "samples.csv",
            "predictions.npy",
            "predictions.csv",
        ]
        # Generative output filenames must list samples.* before predictions.*
        # (samples.* are canonical; predictions.* are legacy aliases).
        outputs = payload["expected_agent_output_filenames"]
        assert outputs.index("samples.npy") < outputs.index("predictions.npy")


class TestWriteTaskManifestRejectsUnsupported:
    @pytest.mark.parametrize("im", ["trading", "gym", "realtime_forecasting"])
    def test_rejects_non_forecasting_generative(self, tmp_path, im):
        with pytest.raises(ValueError, match="forecasting.*generative"):
            _write_task_manifest(tmp_path, _make_candidate(), im, [])


# _stage_static_test_files


class TestStageStaticTestFiles:
    def test_forecasting_stages_correct_file(self, tmp_path):
        _stage_static_test_files(tmp_path, "forecasting")
        staged = tmp_path / "test_task.py"
        helpers = tmp_path / "_test_helpers.py"
        assert staged.exists() and helpers.exists()
        # The staged test file must be the forecasting variant verbatim.
        repo_src = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "openfinai_pipeline"
            / "benchmark"
            / "static_tests"
            / "test_task_forecasting.py"
        )
        assert staged.read_bytes() == repo_src.read_bytes()

    def test_generative_stages_correct_file(self, tmp_path):
        _stage_static_test_files(tmp_path, "generative")
        staged = tmp_path / "test_task.py"
        repo_src = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "openfinai_pipeline"
            / "benchmark"
            / "static_tests"
            / "test_task_generative.py"
        )
        assert staged.read_bytes() == repo_src.read_bytes()

    def test_rejects_unsupported_interaction_model(self, tmp_path):
        with pytest.raises(ValueError, match="forecasting.*generative"):
            _stage_static_test_files(tmp_path, "trading")


# Static-tests directory must exist with the expected files


class TestStaticTestsPackageLayout:
    @pytest.fixture(scope="class")
    def static_tests_dir(self) -> Path:
        return (
            Path(__file__).resolve().parents[1]
            / "src"
            / "openfinai_pipeline"
            / "benchmark"
            / "static_tests"
        )

    def test_package_init_exists(self, static_tests_dir):
        assert (static_tests_dir / "__init__.py").exists()

    def test_helpers_module_exists(self, static_tests_dir):
        assert (static_tests_dir / "_test_helpers.py").exists()

    def test_forecasting_test_file_exists(self, static_tests_dir):
        assert (static_tests_dir / "test_task_forecasting.py").exists()

    def test_generative_test_file_exists(self, static_tests_dir):
        assert (static_tests_dir / "test_task_generative.py").exists()


# End-to-end: synthesize a tiny bundle and run pytest like Sandbox.run_tests


_FORECAST_TASK_PY = '''\
from typing import Any, Dict, List, Optional
import numpy as np
from openfinai_pipeline.benchmark.contracts import ForecastingTask, TaskMetadata


class DemoTask(ForecastingTask):
    """Tiny synthetic task used by the harness end-to-end test.

    Uses the B-shape ``self._data`` contract the install slicer
    materialises: ``{"train": {...}, "test": {...}}``. ``get_features``
    returns test features; ``get_train_features`` /
    ``get_train_ground_truth`` are inherited from the base.
    ``get_ground_truth`` remains inherited so held-out targets stay outside
    agent-side code.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    def metadata(self) -> TaskMetadata:
        return TaskMetadata(
            task_id="demo_task",
            title="Demo Task",
            description="Synthetic task for harness E2E testing.",
            interaction_model="forecasting",
            data_requirements=["synthetic_demo"],
            difficulty="easy",
            version="0.1.0",
        )

    def get_observation_space(self) -> Dict[str, Any]:
        return {"shape": (3,), "dtype": "float32"}

    def get_action_space(self) -> Dict[str, Any]:
        return {"type": "continuous", "shape": (1,), "low": -10.0, "high": 10.0}

    def get_features(self) -> Any:
        # Test features only — the agent's prediction inputs.
        return np.asarray(self._data["test"]["features"], dtype=np.float32)
'''


_FORECAST_EVALUATOR_PY = '''\
"""Synthetic evaluator with prediction-sensitive mockmetric and reward keys."""

from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np

from openfinai_pipeline.benchmark.contracts import BaseEvaluator


class DemoTaskEvaluator(BaseEvaluator):
    REWARD_NAMES: List[str] = ["mockmetric"]

    def score(
        self,
        predictions: Any,
        ground_truth: Any,
        weights: Optional[List[float]] = None,
        reward_output: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        pred = np.asarray(predictions, dtype=np.float64)
        gt = np.asarray(ground_truth, dtype=np.float64)
        n = min(len(pred), len(gt))
        if n == 0:
            mse = 0.0
        else:
            mse = float(np.mean((pred[:n] - gt[:n]) ** 2))
        scores: Dict[str, float] = {
            "mockmetric": -mse,
            "reward": -mse,
        }
        return scores
'''


_FORECAST_INSTRUCTION_MD = """# Demo Task

Write your predictions to `predictions.npy` (or `predictions.csv`) in the
agent output directory. The evaluator at `/data/evaluator.py` scores them
via `score(predictions, ground_truth, ...)`.
"""


@pytest.fixture
def synthesized_forecasting_bundle(tmp_path) -> Path:
    """Hand-build a tiny task package that satisfies the static tests."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "task.py").write_text(_FORECAST_TASK_PY, encoding="utf-8")
    (bundle / "evaluator.py").write_text(_FORECAST_EVALUATOR_PY, encoding="utf-8")
    (bundle / "instruction.md").write_text(_FORECAST_INSTRUCTION_MD, encoding="utf-8")
    (bundle / "__init__.py").write_text("", encoding="utf-8")
    _stage_static_test_files(bundle, "forecasting")
    _write_task_manifest(
        bundle,
        _make_candidate(task_id="demo_task", scope_id="demo_scope"),
        "forecasting",
        ["MockMetric"],
    )
    return bundle


class TestStaticTestsAgainstSynthesizedBundle:
    def test_static_tests_pass(self, synthesized_forecasting_bundle):
        """Spawn pytest like Sandbox.run_tests and assert green."""
        bundle = synthesized_forecasting_bundle

        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        path_entries = [
            str(repo_root),
            str(repo_root / "src"),
            str(repo_root / "data" / "knowledge_base" / "rewards"),
        ]
        existing = env.get("PYTHONPATH", "")
        if existing:
            path_entries.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(path_entries)

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "test_task.py",
            "-v",
            "--tb=short",
            "--no-header",
        ]
        result = subprocess.run(
            cmd,
            cwd=str(bundle),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Static tests failed against synthesized bundle.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
