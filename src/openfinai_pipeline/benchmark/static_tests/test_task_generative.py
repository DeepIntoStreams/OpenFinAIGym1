"""
Deterministic Phase 4 tests for generative tasks.

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

from openfinai_pipeline.benchmark.contracts import BaseTask, GenerativeTask

from _test_helpers import (
    build_generative_fixture,
    import_evaluator_class,
    import_task_class,
    load_manifest,
    make_generative_task,
    wire_evaluator,
)


# Module-scoped fixtures


@pytest.fixture(scope="module")
def manifest() -> Dict[str, Any]:
    return load_manifest()


@pytest.fixture(scope="module")
def fixture_data() -> Dict[str, Any]:
    return build_generative_fixture()


@pytest.fixture
def task(manifest: Dict[str, Any], fixture_data: Dict[str, Any]):
    return make_generative_task(manifest, fixture_data)


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
    def test_task_is_generative_subclass(self, manifest):
        cls = import_task_class(manifest)
        assert issubclass(cls, GenerativeTask), (
            f"{manifest['class_name']} must subclass GenerativeTask "
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
        ref = fixture_data["test"]["ground_truth"]
        zeros = np.zeros_like(ref)
        scores = evaluator.score(zeros, ref)
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
        ref = fixture_data["test"]["ground_truth"]
        zeros = np.zeros_like(ref)
        scores = evaluator.score(zeros, ref)
        for key in manifest["expected_score_keys"]:
            value = scores.get(key)
            assert isinstance(value, (int, float)), (
                f"score[{key!r}]={value!r} (type {type(value).__name__}) "
                "must be numeric"
            )

    def test_reward_sentinel_is_finite(self, evaluator, fixture_data):
        ref = fixture_data["test"]["ground_truth"]
        zeros = np.zeros_like(ref)
        scores = evaluator.score(zeros, ref)
        assert "reward" in scores, "score() must always emit a 'reward' key"
        assert np.isfinite(scores["reward"]), (
            f"reward={scores['reward']} is non-finite"
        )


class TestT9_ScoreDeterministic:
    def test_score_is_deterministic(self, manifest, fixture_data):
        ref = fixture_data["test"]["ground_truth"]
        zeros = np.zeros_like(ref)
        ev_cls = import_evaluator_class(manifest)
        scores_a = ev_cls().score(zeros, ref)
        scores_b = ev_cls().score(zeros, ref)
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


# Generative-only tests (G1 .. G4)


class TestG1_GetTrainReferenceData:
    """Replaces the legacy get_reference_data check.

    The held-out test reference is verifier-only; the agent uses
    ``get_train_reference_data`` to fit a generator. For the
    unconditional case (no held-out target), the task overrides
    ``get_reference_data`` to return the full reference and that is
    the exception path checked separately.
    """

    def test_get_train_reference_data_returns_array_like(self, task):
        ref = task.get_train_reference_data()
        assert ref is not None
        assert hasattr(ref, "__len__")
        assert len(ref) > 0


class TestG2_GetConditioningOptional:
    def test_get_conditioning_data_is_none_or_array_like(self, task):
        cond = task.get_conditioning_data()
        if cond is None:
            return
        assert hasattr(cond, "__len__")


class TestG3_GenerateAndEvaluateKeys:
    def test_generate_and_evaluate_train_emits_expected_keys(
        self, wired_task, manifest
    ):
        """Score generated samples against train reference (agent-side path).

        The conditional-generative test reference is held out; the agent
        scores against the train reference for sanity-checking. The
        unconditional case goes through the same path because
        ``get_train_reference_data`` falls back to ``self._data["reference"]``.
        """
        ref = wired_task.get_train_reference_data()
        zeros = np.zeros_like(ref)
        scores = wired_task.generate_and_evaluate(zeros, split="train")
        assert isinstance(scores, dict)
        actual_keys = {k for k in scores.keys() if not k.startswith("_")}
        expected = set(manifest["expected_score_keys"])
        missing = expected - actual_keys
        assert not missing, (
            f"generate_and_evaluate is missing expected keys: "
            f"{sorted(missing)}; got: {sorted(actual_keys)}"
        )

    def test_generate_and_evaluate_test_split_raises(self, wired_task):
        """split='test' is verifier-only; agent-side calls must raise."""
        ref = wired_task.get_train_reference_data()
        zeros = np.zeros_like(ref)
        try:
            wired_task.generate_and_evaluate(zeros, split="test")
        except PermissionError:
            return
        pytest.fail(
            "generate_and_evaluate(split='test') did not raise PermissionError"
        )


class TestG4_AntiDegenerate:
    def test_zeros_and_reference_samples_score_differently(self, wired_task):
        ref = wired_task.get_train_reference_data()
        ref_arr = np.asarray(ref, dtype=np.float64)
        zeros = np.zeros_like(ref_arr)
        copy = ref_arr.copy()

        zero_score = wired_task.generate_and_evaluate(zeros, split="train")
        ref_score = wired_task.generate_and_evaluate(copy, split="train")

        z = float(zero_score.get("reward", 0.0))
        r = float(ref_score.get("reward", 0.0))
        assert abs(z - r) > 1e-9, (
            "generate_and_evaluate(zeros) and generate_and_evaluate(reference) "
            f"produced identical 'reward' values ({z} vs {r}). The configured "
            "metrics may ignore generated samples, or the evaluator dispatch "
            "is broken. If this task's rewards are genuinely sample-"
            "independent, curate the reward set."
        )
