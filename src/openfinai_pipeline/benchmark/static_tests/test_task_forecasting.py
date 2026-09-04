"""
Deterministic Phase 4 tests for forecasting tasks.

Staged verbatim into the task package by ``write_deterministic_harbor_assets``.
All per-task assertion values come from ``manifest.json`` (sibling file).
Tests inject a synthetic fixture via ``task._data`` -- they do NOT exercise
the real bundled ``load.py`` (the post-install smoke gate covers that).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from openfinai_pipeline.benchmark.contracts import BaseTask, ForecastingTask

from _test_helpers import (
    build_forecasting_fixture,
    import_evaluator_class,
    import_task_class,
    load_manifest,
    make_forecasting_task,
    wire_evaluator,
)


# Module-scoped fixtures


@pytest.fixture(scope="module")
def manifest() -> Dict[str, Any]:
    return load_manifest()


@pytest.fixture(scope="module")
def fixture_data() -> Dict[str, Any]:
    return build_forecasting_fixture()


@pytest.fixture
def task(manifest: Dict[str, Any], fixture_data: Dict[str, Any]):
    return make_forecasting_task(manifest, fixture_data)


@pytest.fixture
def wired_task(task, manifest: Dict[str, Any]):
    wire_evaluator(task, manifest)
    return task


@pytest.fixture(scope="module")
def evaluator(manifest: Dict[str, Any]):
    return import_evaluator_class(manifest)()


# Universal tests (T1 .. T11)


class TestT1_TaskModuleImports:
    def test_task_class_is_importable(self, manifest):
        cls = import_task_class(manifest)
        assert isinstance(cls, type)


class TestT2_TaskSubclass:
    def test_task_is_forecasting_subclass(self, manifest):
        cls = import_task_class(manifest)
        assert issubclass(cls, ForecastingTask), (
            f"{manifest['class_name']} must subclass ForecastingTask "
            f"(interaction_model={manifest['interaction_model']!r})"
        )


class TestT3_MetadataFields:
    def test_metadata_task_id_matches_manifest(self, task, manifest):
        meta = task.metadata()
        assert meta.task_id == manifest["task_id"], (
            f"metadata.task_id={meta.task_id!r} but manifest.task_id="
            f"{manifest['task_id']!r}"
        )

    def test_metadata_interaction_model_matches_manifest(self, task, manifest):
        meta = task.metadata()
        assert meta.interaction_model == manifest["interaction_model"], (
            f"metadata.interaction_model={meta.interaction_model!r} but "
            f"manifest.interaction_model={manifest['interaction_model']!r}"
        )

    def test_metadata_version_format(self, task):
        meta = task.metadata()
        assert re.match(r"^\d+\.\d+\.\d+$", meta.version), (
            f"version {meta.version!r} must match \\d+\\.\\d+\\.\\d+"
        )

    def test_metadata_required_fields_populated(self, task):
        meta = task.metadata()
        assert meta.title.strip(), "metadata.title is empty"
        assert meta.description.strip(), "metadata.description is empty"
        assert meta.data_requirements, "metadata.data_requirements is empty"
        assert meta.difficulty in ("easy", "medium", "hard")


class TestT4_LoadDataInherited:
    def test_load_data_is_inherited_from_base_task(self, manifest):
        cls = import_task_class(manifest)
        assert cls.load_data is BaseTask.load_data, (
            "task class overrides load_data; the framework owns this method "
            "(it delegates to the bundled load.py). Remove the override."
        )


class TestT5_Spaces:
    def test_observation_space_has_shape(self, task):
        obs_space = task.get_observation_space()
        assert isinstance(obs_space, dict)
        assert "shape" in obs_space, f"observation_space missing 'shape': {obs_space}"

    def test_action_space_has_type(self, task):
        action_space = task.get_action_space()
        assert isinstance(action_space, dict)
        assert action_space.get("type") in ("discrete", "continuous"), (
            f"action_space.type must be 'discrete' or 'continuous': {action_space}"
        )


class TestT6_EvaluatorModuleImports:
    def test_evaluator_class_is_importable(self, manifest):
        cls = import_evaluator_class(manifest)
        assert isinstance(cls, type)


class TestT7_EvaluatorScoreKeys:
    def test_score_returns_expected_keys(self, evaluator, manifest, fixture_data):
        gt = fixture_data["test"]["ground_truth"]
        zeros = np.zeros_like(gt)
        scores = evaluator.score(zeros, gt)
        assert isinstance(scores, dict), f"score() returned {type(scores).__name__}"
        actual_keys = {k for k in scores.keys() if not k.startswith("_")}
        expected = set(manifest["expected_score_keys"])
        missing = expected - actual_keys
        assert not missing, (
            f"score() output is missing expected reward keys: {sorted(missing)}; "
            f"got keys: {sorted(actual_keys)}"
        )


class TestT8_ScoreValuesNumeric:
    def test_score_values_are_numeric(self, evaluator, manifest, fixture_data):
        gt = fixture_data["test"]["ground_truth"]
        zeros = np.zeros_like(gt)
        scores = evaluator.score(zeros, gt)
        for key in manifest["expected_score_keys"]:
            value = scores.get(key)
            assert isinstance(value, (int, float)), (
                f"score[{key!r}]={value!r} (type {type(value).__name__}) "
                "must be numeric"
            )

    def test_reward_sentinel_is_finite(self, evaluator, fixture_data):
        gt = fixture_data["test"]["ground_truth"]
        zeros = np.zeros_like(gt)
        scores = evaluator.score(zeros, gt)
        assert "reward" in scores, "score() must always emit a 'reward' key"
        assert np.isfinite(scores["reward"]), (
            f"reward={scores['reward']} is non-finite"
        )


class TestT9_ScoreDeterministic:
    def test_score_is_deterministic(self, manifest, fixture_data):
        gt = fixture_data["test"]["ground_truth"]
        zeros = np.zeros_like(gt)
        ev_cls = import_evaluator_class(manifest)
        scores_a = ev_cls().score(zeros, gt)
        scores_b = ev_cls().score(zeros, gt)
        common_keys = set(scores_a.keys()) & set(scores_b.keys())
        assert common_keys, "evaluator returned no overlapping keys across runs"
        for key in common_keys:
            if key.startswith("_"):
                continue
            a = scores_a[key]
            b = scores_b[key]
            if isinstance(a, float) and np.isnan(a) and isinstance(b, float) and np.isnan(b):
                continue
            assert a == b, (
                f"non-deterministic score: scores[{key!r}] differs across "
                f"runs: {a!r} vs {b!r}"
            )


class TestT10_TaskPyAstHygiene:
    @pytest.fixture(scope="class")
    def task_source(self) -> str:
        path = Path(__file__).resolve().parent / "task.py"
        return path.read_text(encoding="utf-8")

    def test_no_absolute_paths(self, task_source):
        tree = ast.parse(task_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                # POSIX absolute or Windows drive letter (e.g. "C:/" / "C:\\")
                is_posix_abs = v.startswith("/") and len(v) > 1 and not v.startswith("//")
                is_win_abs = len(v) >= 3 and v[1] == ":" and v[2] in ("/", "\\")
                assert not (is_posix_abs or is_win_abs), (
                    f"absolute path literal at line {node.lineno}: {v!r}"
                )

    def test_no_dangerous_imports(self, task_source):
        tree = ast.parse(task_source)
        blocked = {"importlib"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in blocked, (
                        f"task.py imports blocked module {alias.name!r} at "
                        f"line {node.lineno}"
                    )
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root not in blocked, (
                    f"task.py uses 'from {node.module} import ...' (blocked) "
                    f"at line {node.lineno}"
                )

    def test_no_dynamic_code_execution(self, task_source):
        tree = ast.parse(task_source)
        forbidden_calls = {"eval", "exec", "compile"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls, (
                    f"task.py calls {node.func.id}() at line {node.lineno}"
                )


class TestT11_InstructionMentionsOutputFilenames:
    def test_instruction_md_mentions_canonical_output(self, manifest):
        instr_path = Path(__file__).resolve().parent / "instruction.md"
        if not instr_path.exists():
            pytest.skip("instruction.md not staged alongside test file")
        text = instr_path.read_text(encoding="utf-8")
        candidates = manifest["expected_agent_output_filenames"]
        if not any(name in text for name in candidates):
            pytest.fail(
                f"instruction.md mentions none of {candidates}; agent will "
                "not know the canonical output filenames the evaluator loads."
            )


# Forecasting-only tests (F1 .. F5)


class TestF1_GetFeatures:
    def test_get_features_returns_test_features(self, task):
        """``get_features`` returns the **test** features the agent predicts on."""
        features = task.get_features()
        assert features is not None
        assert hasattr(features, "__len__")
        assert len(features) > 0


class TestF2_GetTrainGroundTruth:
    """Replaces the legacy get_ground_truth check.

    The held-out test ground_truth is intentionally NOT exposed to the
    agent — the default ``get_ground_truth`` raises ``PermissionError``.
    The agent's training loop uses ``get_train_ground_truth`` to fit
    against, so that's what this test validates.
    """

    def test_get_train_ground_truth_returns_array_like(self, task):
        gt = task.get_train_ground_truth()
        assert gt is not None
        assert hasattr(gt, "__len__")
        assert len(gt) > 0

    def test_get_ground_truth_raises_permission_error(self, task):
        """The default ``get_ground_truth`` keeps held-out targets verifier-side.

        Auto-generated forecasting tasks must NOT override
        ``get_ground_truth``; calling it raises ``PermissionError`` so
        an agent cannot read the held-out test target through the
        framework. The verifier reads test GT from
        ``/eval-data/test_ground_truth.h5`` instead.
        """
        try:
            task.get_ground_truth()
        except PermissionError:
            return
        # If we got here, the task overrode the default; that's the
        # legacy curated-task path and should not appear in
        # auto-generated tasks.
        import pytest

        pytest.fail(
            "get_ground_truth() did not raise PermissionError. "
            "Auto-generated tasks must inherit the base ForecastingTask "
            "default; only curated tasks may override it."
        )


class TestF3_TrainFeaturesGroundTruthAligned:
    def test_train_features_and_ground_truth_have_same_first_dim(self, task):
        features = task.get_train_features()
        gt = task.get_train_ground_truth()
        f_len = features.shape[0] if hasattr(features, "shape") else len(features)
        g_len = gt.shape[0] if hasattr(gt, "shape") else len(gt)
        assert f_len == g_len, (
            f"train features ({f_len}) and train ground_truth ({g_len}) "
            "sample counts must align (off-by-one between feature window "
            "and label is a common bug)"
        )


class TestF4_PredictAndEvaluateKeys:
    def test_predict_and_evaluate_train_emits_expected_keys(
        self, wired_task, manifest
    ):
        """Score against the train split (agent-side path).

        ``split='test'`` is verifier-only and raises; ``split='train'``
        scores against the train ground_truth which the agent legitimately
        has access to.
        """
        gt = wired_task.get_train_ground_truth()
        zeros = np.zeros_like(gt)
        scores = wired_task.predict_and_evaluate(zeros, split="train")
        assert isinstance(scores, dict)
        actual_keys = {k for k in scores.keys() if not k.startswith("_")}
        expected = set(manifest["expected_score_keys"])
        missing = expected - actual_keys
        assert not missing, (
            f"predict_and_evaluate is missing expected keys: "
            f"{sorted(missing)}; got: {sorted(actual_keys)}"
        )

    def test_predict_and_evaluate_test_split_raises(self, wired_task):
        """split='test' is verifier-only; agent-side calls must raise."""
        gt = wired_task.get_train_ground_truth()
        zeros = np.zeros_like(gt)
        try:
            wired_task.predict_and_evaluate(zeros, split="test")
        except PermissionError:
            return
        import pytest

        pytest.fail(
            "predict_and_evaluate(split='test') did not raise PermissionError "
            "— this leaks held-out data; verifier-only scoring must "
            "not be reachable from agent-side code."
        )


class TestF5_AntiDegenerate:
    def test_zeros_and_perfect_predictions_score_differently(self, wired_task):
        gt = wired_task.get_train_ground_truth()
        gt_arr = np.asarray(gt, dtype=np.float64)
        zeros = np.zeros_like(gt_arr)
        perfect = gt_arr.copy()

        zero_score = wired_task.predict_and_evaluate(zeros, split="train")
        perfect_score = wired_task.predict_and_evaluate(perfect, split="train")

        z = float(zero_score.get("reward", 0.0))
        p = float(perfect_score.get("reward", 0.0))
        assert abs(z - p) > 1e-9, (
            "predict_and_evaluate(zeros) and predict_and_evaluate(train_gt) "
            f"produced identical 'reward' values ({z} vs {p}). The configured "
            "metrics may ignore predictions, or the evaluator dispatch is "
            "broken. If this task's rewards are genuinely prediction-"
            "independent (e.g. coverage-only), curate the reward set."
        )
