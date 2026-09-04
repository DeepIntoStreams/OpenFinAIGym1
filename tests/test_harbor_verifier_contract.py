from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from openfinai_harbor.eval import run_evaluation_explicit
from openfinai_pipeline.benchmark.builders.task_builder import assemble_evaluator


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_test_gt_h5(eval_data_dir: Path, gt: np.ndarray | dict[str, np.ndarray]) -> None:
    """Write a synthetic ``test_ground_truth.h5`` mirroring the install slicer.

    Single-symbol ndarrays go under ``/ground_truth/_value`` (the
    sentinel key the evaluator's reader unwraps); dict-of-ndarrays
    panel inputs become one dataset per symbol under ``/ground_truth/``.
    """
    eval_data_dir.mkdir(parents=True, exist_ok=True)
    out = eval_data_dir / "test_ground_truth.h5"
    with h5py.File(out, "w") as f:
        gt_group = f.create_group("ground_truth")
        if isinstance(gt, np.ndarray):
            gt_group.create_dataset(
                "_value", data=np.asarray(gt, dtype=np.float64),
                compression="gzip", shuffle=True,
            )
        else:
            for sym, arr in gt.items():
                gt_group.create_dataset(
                    sym, data=np.asarray(arr, dtype=np.float64),
                    compression="gzip", shuffle=True,
                )


def test_run_evaluation_defers_missing_io_to_evaluator(tmp_path, monkeypatch):
    evaluator_path = tmp_path / "evaluator.py"
    evaluator_path.write_text(
        """
import json
from pathlib import Path


class RuntimeOwnedEvaluator:
    REWARD_NAMES = ["reward"]
    EXPECTS_PREDICTIONS = False
    EXPECTS_GROUND_TRUTH = False

    def score(self, predictions, ground_truth, weights=None, reward_output=None, **kwargs):
        agent_output_dir = Path(kwargs["agent_output_dir"])
        data_dir = Path(kwargs["data_dir"])
        assert predictions is None
        assert ground_truth is None
        assert (agent_output_dir / "agent_artifact.txt").exists()
        assert (data_dir / "payload.csv").exists()
        payload = {"reward": 0.5}
        Path(reward_output).write_text(json.dumps(payload), encoding="utf-8")
        return payload
""".strip()
        + "\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "payload.csv").write_text("value\n1\n2\n", encoding="utf-8")
    train_py = tmp_path / "train.py"
    train_py.write_text("print('train')\n", encoding="utf-8")
    work_dir = tmp_path / "agent_work"
    work_dir.mkdir()
    (work_dir / "agent_artifact.txt").write_text("ok\n", encoding="utf-8")
    reward_output = tmp_path / "reward.json"

    monkeypatch.setattr(run_evaluation_explicit, "_find_train_py", lambda: train_py)
    monkeypatch.setattr(run_evaluation_explicit, "_run_train", lambda *_args, **_kwargs: (work_dir, True))
    monkeypatch.setattr(run_evaluation_explicit, "_load_predictions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_evaluation_explicit, "_load_ground_truth", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_evaluation_explicit.py",
            "--evaluator-path",
            str(evaluator_path),
            "--data-dir",
            str(data_dir),
            "--reward-output",
            str(reward_output),
        ],
    )

    run_evaluation_explicit.main()

    payload = json.loads(reward_output.read_text(encoding="utf-8"))
    assert payload == {"reward": 0.5}


def test_assembled_forecasting_evaluator_loads_payload_and_predictions_csv(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    rewards_root = repo_root / "data" / "knowledge_base" / "rewards"
    staging_dir = tmp_path / "forecast_eval"
    assemble_evaluator(
        task_id="forecast_contract_task",
        task_title="Forecast Contract Task",
        rewards=[
            {
                "name": "MSE",
                "description": "Mean squared error between predictions and targets.",
                "reward_script_path": str((rewards_root / "reward_bank.py").resolve()),
                "resolved_class_name": "MSELoss",
            }
        ],
        eval_root=rewards_root,
        staging_dir=staging_dir,
        interaction_model="forecasting",
        artifact_plan={
            "dataset_payload_files": ["labels.csv"],
            "expected_agent_outputs": ["predictions.csv"],
            "ground_truth_artifacts": [
                {"role": "ground_truth", "path": "labels.csv", "format": "csv"}
            ],
            "reference_artifacts": [],
            "feature_artifacts": [],
            "has_ground_truth": True,
            "manifest_status": "labeled",
            "interaction_model": "forecasting",
        },
    )
    module = _load_module("forecast_contract_evaluator", staging_dir / "evaluator.py")
    evaluator = module.ForecastContractTaskEvaluator()

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "labels.csv").write_text("value\n1\n2\n3\n", encoding="utf-8")
    # The verifier reads the held-out test target from
    # ``<eval_data_dir>/test_ground_truth.h5`` — a separate mount the
    # agent doesn't see. Mirror the install layout by writing it to
    # ``tmp_path/eval-data/`` so the evaluator's
    # ``data_dir/../eval-data`` fallback finds it.
    _write_test_gt_h5(tmp_path / "eval-data", np.array([1.0, 2.0, 3.0]))
    agent_output_dir = tmp_path / "agent"
    agent_output_dir.mkdir()
    (agent_output_dir / "predictions.csv").write_text("value\n1\n2\n4\n", encoding="utf-8")
    reward_output = tmp_path / "forecast_reward.json"

    scores = evaluator.score(
        predictions=None,
        ground_truth=None,
        reward_output=reward_output,
        agent_output_dir=agent_output_dir,
        data_dir=data_dir,
    )

    # MSE → MSELoss bank class → "mseloss" canonical key.
    assert "mseloss" in scores
    assert np.isfinite(scores["mseloss"])
    assert np.isclose(scores["mseloss"], 1.0 / 3.0)
    assert scores["reward"] == scores["mseloss"]
    assert "_error" not in scores
    assert reward_output.exists()


def test_assembled_generative_evaluator_prefers_samples_npy(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    rewards_root = repo_root / "data" / "knowledge_base" / "rewards"
    staging_dir = tmp_path / "generative_eval"
    assemble_evaluator(
        task_id="generative_contract_task",
        task_title="Generative Contract Task",
        rewards=[
            {
                "name": "MeanAbsDiffReward",
                "description": "Computes average L1 distance between real and generated samples.",
                "reward_script_path": str((rewards_root / "reward_bank.py").resolve()),
                "resolved_class_name": "MeanLoss",
            }
        ],
        eval_root=rewards_root,
        staging_dir=staging_dir,
        interaction_model="generative",
        artifact_plan={
            "dataset_payload_files": ["test_sequences.npy", "train_sequences.npy"],
            "expected_agent_outputs": [
                "samples.npy",
                "samples.csv",
                "predictions.npy",
                "predictions.csv",
            ],
            "ground_truth_artifacts": [],
            "reference_artifacts": [
                {
                    "role": "reference",
                    "path": "test_sequences.npy",
                    "format": "npy",
                }
            ],
            "feature_artifacts": [
                {
                    "role": "features",
                    "path": "train_sequences.npy",
                    "format": "npy",
                }
            ],
            "has_ground_truth": True,
            "manifest_status": "labeled",
            "interaction_model": "generative",
        },
    )
    module = _load_module("generative_contract_evaluator", staging_dir / "evaluator.py")
    evaluator = module.GenerativeContractTaskEvaluator()

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.save(data_dir / "train_sequences.npy", np.ones((2, 3, 1), dtype=np.float32))
    np.save(data_dir / "test_sequences.npy", np.ones((2, 3, 1), dtype=np.float32))
    # Generative test reference is held out for conditional generative;
    # the verifier reads it from eval-data/test_ground_truth.h5.
    _write_test_gt_h5(
        tmp_path / "eval-data",
        np.ones((2, 3, 1), dtype=np.float64),
    )
    agent_output_dir = tmp_path / "agent"
    agent_output_dir.mkdir()
    np.save(agent_output_dir / "samples.npy", np.zeros((2, 3, 1), dtype=np.float32))
    reward_output = tmp_path / "generative_reward.json"

    scores = evaluator.score(
        predictions=None,
        ground_truth=None,
        reward_output=reward_output,
        agent_output_dir=agent_output_dir,
        data_dir=data_dir,
    )

    metric_keys = [key for key in scores if key not in {"reward", "_error"}]
    assert metric_keys
    assert np.isfinite(scores["reward"])
    assert scores["reward"] > 0.0
    assert "_error" not in scores
    assert reward_output.exists()


def test_assembled_generative_pair_evaluator_uses_real_fake_order(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    rewards_root = repo_root / "data" / "knowledge_base" / "rewards"
    staging_dir = tmp_path / "generative_pair_eval"
    assemble_evaluator(
        task_id="generative_pair_contract_task",
        task_title="Generative Pair Contract Task",
        rewards=[
            {
                "name": "ONNDReward",
                "description": "Outgoing Nearest Neighbour Distance.",
                "reward_script_path": str((rewards_root / "reward_bank.py").resolve()),
                "resolved_class_name": "ONNDMetricLoss",
            },
            {
                "name": "VARReward",
                "description": "Value-at-risk style discrepancy.",
                "reward_script_path": str((rewards_root / "reward_bank.py").resolve()),
                "resolved_class_name": "VARMetricLoss",
            },
        ],
        eval_root=rewards_root,
        staging_dir=staging_dir,
        interaction_model="generative",
        dataset_payload_files=["reference.npy"],
        expected_agent_outputs=["samples.npy"],
    )
    module = _load_module("generative_pair_contract_evaluator", staging_dir / "evaluator.py")
    evaluator = module.GenerativePairContractTaskEvaluator()

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.save(data_dir / "reference.npy", np.ones((4, 3, 2), dtype=np.float32))
    # Held-out reference for the generative pair tasks lives in the
    # verifier-only mount at <tmp>/eval-data/test_ground_truth.h5.
    _write_test_gt_h5(
        tmp_path / "eval-data",
        np.ones((4, 3, 2), dtype=np.float64),
    )
    agent_output_dir = tmp_path / "agent"
    agent_output_dir.mkdir()
    np.save(agent_output_dir / "samples.npy", np.zeros((4, 3, 2), dtype=np.float32))

    scores = evaluator.score(
        predictions=None,
        ground_truth=None,
        agent_output_dir=agent_output_dir,
        data_dir=data_dir,
    )

    # ONNDReward → ONNDMetricLoss class → "onndmetricloss" key;
    # VARReward → VARMetricLoss class → "varmetricloss" key.
    assert "onndmetricloss" in scores
    assert "varmetricloss" in scores
    assert np.isfinite(scores["onndmetricloss"])
    assert np.isfinite(scores["varmetricloss"])
    assert np.isfinite(scores["reward"])
    assert "_error" not in scores


def test_task_template_wires_sibling_evaluator_with_safe_fallback():
    template_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "openfinai_pipeline"
        / "benchmark"
        / "templates"
        / "task.py.j2"
    )
    source = template_path.read_text(encoding="utf-8")
    assert "from evaluator import" in source
    assert "except Exception:" in source


def test_static_tests_inject_fixture_and_call_batch_apis():
    """Static tests inject fixtures and exercise each batch API."""
    static_dir = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "openfinai_pipeline"
        / "benchmark"
        / "static_tests"
    )
    forecasting_src = (static_dir / "test_task_forecasting.py").read_text(encoding="utf-8")
    generative_src = (static_dir / "test_task_generative.py").read_text(encoding="utf-8")
    helpers_src = (static_dir / "_test_helpers.py").read_text(encoding="utf-8")

    assert "task._data = fixture" in helpers_src

    assert "predict_and_evaluate" in forecasting_src

    assert "generate_and_evaluate" in generative_src

    assert "anti" in forecasting_src.lower() or "anti" in generative_src.lower(), (
        "Static tests should include an anti-degenerate metric check"
    )
