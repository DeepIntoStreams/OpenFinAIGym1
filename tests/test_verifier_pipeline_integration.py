"""Verifier and generated-task integration contracts."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from openfinai_pipeline.benchmark.builders import task_builder


# 1. Dockerfile no longer bakes eval-data into the agent image


def test_dockerfile_is_data_free():
    """Task data is mounted at runtime, never baked into the image."""
    df = task_builder._render_harbor_dockerfile()
    # No legacy eval-data COPY.
    assert "COPY environment/eval-data" not in df
    # No COPY data/ either — data is now mounted, not baked.
    assert "COPY data/" not in df
    # Reject any environment-relative COPY path.
    assert "COPY environment/" not in df
    # The FROM line points at the shared registry-published base.
    assert "FROM nihao0630/openfinai-base:v1" in df
    # WORKDIR still set.
    assert "WORKDIR /workspace" in df


# 2. install_task ships harbor_submit.py into environment/data/


def test_harbor_submit_source_path_resolves_to_real_file():
    p = task_builder._harbor_submit_source_path()
    assert p.exists(), f"expected harbor_submit source at {p}"
    text = p.read_text(encoding="utf-8")
    # Sanity: contains the public submit() function + persist helper.
    assert "def submit(" in text
    assert "def submit_and_persist(" in text
    assert "VERIFIER_URL" in text


def test_install_task_ships_harbor_submit(tmp_path, monkeypatch):
    """End-to-end: build a minimal staging layout, run install_task,
    inspect the resulting bundle for harbor_submit.py."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    (staging_dir / "instruction.md").write_text("# Synthetic task\n")
    (staging_dir / "task.toml").write_text(
        '[metadata]\nname = "scope/synthetic_task"\n'
    )
    (staging_dir / "tests").mkdir()
    (staging_dir / "tests" / "test.sh").write_text("#!/bin/bash\necho hi\n")
    # Minimal evaluator + task.py + manifest.
    (staging_dir / "evaluator.py").write_text("class FakeEvaluator: pass\n")
    (staging_dir / "task.py").write_text("class FakeTask: pass\n")
    (staging_dir / "manifest.json").write_text('{"task_id":"synthetic_task"}\n')
    (staging_dir / "__init__.py").write_text("")

    tasks_root = tmp_path / "tasks"
    datasets_root = tmp_path / "datasets"
    curated_root = tmp_path / "curated"

    dest = task_builder.install_task(
        scope_id="scope",
        task_id="synthetic_task",
        staging_dir=staging_dir,
        tasks_root=tasks_root,
        phase_artifacts=None,
        datasets_root=datasets_root,
        curated_datasets_root=curated_root,
        loader_ready=False,
    )
    bundle_data = dest / "environment" / "data"
    assert (bundle_data / "harbor_submit.py").exists(), (
        "install_task did not ship harbor_submit.py into the bundle"
    )
    # Sanity-check it's the one we expect (not stale).
    assert "submit_and_persist" in (
        bundle_data / "harbor_submit.py"
    ).read_text(encoding="utf-8")
    # The runner is also there.
    assert (bundle_data / "run_evaluation_explicit.py").exists()
    # Dockerfile is rendered without the eval-data COPY.
    df = (dest / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY environment/eval-data" not in df


# 3. run_evaluation_explicit.py branches on VERIFIER_URL


@pytest.fixture
def runner_module():
    """Import a fresh copy of the runner so env-driven branching is testable."""
    import importlib
    import openfinai_harbor.eval.run_evaluation_explicit as mod

    # Reload to pick up any env that the test may have set BEFORE import.
    return importlib.reload(mod)


def test_runner_verifier_mode_skips_local_evaluator(runner_module, tmp_path, monkeypatch):
    """When VERIFIER_URL is set, main() should NOT load the local evaluator
    — just run train.py and check for reward.json."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "harbor_submit.py").write_text("# stub\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # train.py just writes reward.json to HARBOR_REWARD_OUTPUT.
    train_py = workspace / "train.py"
    train_py.write_text(
        "import json, os\n"
        "out = os.environ['HARBOR_REWARD_OUTPUT']\n"
        "os.makedirs(os.path.dirname(out), exist_ok=True)\n"
        "with open(out, 'w') as f:\n"
        "    json.dump({'reward': 0.5, 'mse': 0.25}, f)\n"
    )

    reward_path = tmp_path / "logs" / "verifier" / "reward.json"
    monkeypatch.setenv("VERIFIER_URL", "http://localhost:5555")
    monkeypatch.setattr(
        runner_module, "_find_train_py", lambda: train_py
    )
    # _load_evaluator should NOT be called in verifier mode — make it a
    # tripwire that fails the test loudly if the wrong branch is taken.
    monkeypatch.setattr(
        runner_module,
        "_load_evaluator",
        lambda *a, **kw: pytest.fail(
            "_load_evaluator was called in verifier mode (wrong branch)"
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_evaluation_explicit.py",
            "--evaluator-path",
            str(data_dir / "evaluator.py"),
            "--data-dir",
            str(data_dir),
            "--reward-output",
            str(reward_path),
        ],
    )
    runner_module.main()

    assert reward_path.exists()
    import json
    body = json.loads(reward_path.read_text(encoding="utf-8"))
    # New schema: reward.json is strict single-key ({"reward": <float>}).
    # The runner normalises a legacy multi-key agent write by splitting it
    # into single-key reward.json + multi-key metrics.json sidecar so
    # harbor's Mean.compute is happy regardless of which write path the
    # agent took.
    assert body == {"reward": 0.5}
    metrics_path = reward_path.parent / "metrics.json"
    assert metrics_path.exists(), "metrics.json sidecar must accompany reward.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["reward"] == 0.5
    assert metrics["mse"] == 0.25


def test_runner_verifier_mode_writes_zero_reward_when_train_omits_persist(
    runner_module, tmp_path, monkeypatch
):
    """If the agent's train.py does NOT write reward.json, runner should
    emit zero with a clear error — not silently succeed."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Agent that exits 0 but writes NO reward.json
    train_py = workspace / "train.py"
    train_py.write_text("print('I forgot to call submit_and_persist')\n")

    reward_path = tmp_path / "logs" / "reward.json"
    monkeypatch.setenv("VERIFIER_URL", "http://localhost:5555")
    monkeypatch.setattr(runner_module, "_find_train_py", lambda: train_py)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_evaluation_explicit.py",
            "--evaluator-path",
            str(data_dir / "evaluator.py"),
            "--data-dir",
            str(data_dir),
            "--reward-output",
            str(reward_path),
        ],
    )
    runner_module.main()

    assert reward_path.exists()
    import json
    body = json.loads(reward_path.read_text(encoding="utf-8"))
    assert body["reward"] == 0.0
    # reward.json is harbor-schema-clean: numeric values only. Structured
    # error info goes to a sibling error.txt — see the Bug 2 fix
    # (harbor's VerifierResult.rewards rejects non-numeric values during
    # pydantic validation, so an `_error` string in reward.json crashes
    # the trial).
    assert "_error" not in body, (
        "reward.json must not contain non-numeric fields; "
        "structured error info belongs in error.txt"
    )
    error_path = reward_path.parent / "error.txt"
    assert error_path.exists(), (
        f"expected sidecar at {error_path} when train.py omits persist"
    )
    assert "submit_and_persist" in error_path.read_text(encoding="utf-8")
