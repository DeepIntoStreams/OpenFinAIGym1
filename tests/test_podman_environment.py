"""Podman command-shape tests using a mocked subprocess layer."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from openfinai_harbor.environments.podman import (
    PodmanEnvironment,
    _sanitize_container_name,
)


# Test scaffolding


def _make_env(
    tmp_path: Path,
    *,
    session_id: str = "trial-abc",
    docker_image: str | None = "nihao0630/openfinai-base:v1",
    mounts: list[dict] | None = None,
    podman_storage_root: str | None = "/workspace/podman_storage",
    podman_run_user: str | int | None = None,
    cpus: int = 2,
    memory_mb: int = 4096,
) -> PodmanEnvironment:
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    if not docker_image:
        (env_dir / "Dockerfile").write_text("FROM alpine:latest\n")
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()
    return PodmanEnvironment(
        environment_dir=env_dir,
        environment_name="test",
        session_id=session_id,
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            docker_image=docker_image,
            cpus=cpus,
            memory_mb=memory_mb,
        ),
        mounts=mounts,
        podman_storage_root=podman_storage_root,
        podman_run_user=podman_run_user,
    )


def _make_fake_proc(
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int = 0,
):
    """Fake asyncio subprocess that records ``communicate`` results."""
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.returncode = returncode
    return proc


@pytest.fixture
def captured_argv():
    """Yield a list that captures every podman argv created."""
    argvs: list[list[str]] = []

    async def _fake_create_subprocess_exec(*args, **kwargs):
        argvs.append(list(args))
        return _make_fake_proc()

    with patch(
        "openfinai_harbor.environments.podman.asyncio.create_subprocess_exec",
        side_effect=_fake_create_subprocess_exec,
    ):
        yield argvs


# Static sanity


def test_type_is_podman():
    assert PodmanEnvironment.type() == "podman"


def test_capabilities_mounted_true():
    inst = PodmanEnvironment.__new__(PodmanEnvironment)
    caps = inst.capabilities
    assert caps.mounted is True
    assert caps.gpus is False


@pytest.mark.parametrize(
    "raw,expected_prefix",
    [
        ("ofg-trial-01", "ofg-trial-01"),
        ("trial/with:bad/chars", "trial-with-bad-chars"),
        ("__leading-bad", "h__leading-bad"),
        ("", "h"),
    ],
)
def test_sanitize_container_name(raw, expected_prefix):
    out = _sanitize_container_name(raw)
    assert out.startswith(expected_prefix)
    assert len(out) <= 128


# _validate_definition


def test_validate_definition_passes_with_docker_image(tmp_path):
    env = _make_env(tmp_path)
    assert env.task_env_config.docker_image == "nihao0630/openfinai-base:v1"


def test_validate_definition_passes_with_local_dockerfile(tmp_path):
    env = _make_env(tmp_path, docker_image=None)
    assert (env.environment_dir / "Dockerfile").exists()


def test_validate_definition_fails_with_neither(tmp_path):
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()
    with pytest.raises(FileNotFoundError, match="docker_image"):
        PodmanEnvironment(
            environment_dir=env_dir,
            environment_name="test",
            session_id="s",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(),
        )


# start() — argv composition


def test_start_argv_basic(tmp_path, captured_argv):
    env = _make_env(tmp_path, session_id="ofg-trial-A")
    asyncio.run(env.start(force_build=False))

    assert len(captured_argv) >= 3
    rm_argv, pull_argv, run_argv = captured_argv[0], captured_argv[1], captured_argv[2]

    for argv in (rm_argv, pull_argv, run_argv):
        assert argv[0] == "podman"
        assert "--root" in argv
        assert "/workspace/podman_storage" in argv

    assert "rm" in rm_argv and "--filter" in rm_argv
    label_kv = [a for a in rm_argv if a.startswith("label=harbor-session=")]
    assert label_kv, f"rm argv missing label filter: {rm_argv}"

    assert "pull" in pull_argv
    assert "nihao0630/openfinai-base:v1" in pull_argv

    assert "run" in run_argv
    assert "--detach" in run_argv
    assert "--name" in run_argv
    assert "--network=host" in run_argv
    assert "--label" in run_argv
    assert "--memory" in run_argv and "4096m" in run_argv
    assert "--cpus" in run_argv and "2" in run_argv
    assert "nihao0630/openfinai-base:v1" in run_argv
    assert run_argv[-2:] == ["sleep", "infinity"]


def test_start_translates_bind_mounts_to_v_flags(tmp_path, captured_argv):
    mounts = [
        {"type": "bind", "source": "/host/data", "target": "/data"},
        {"type": "bind", "source": "/host/logs", "target": "/logs", "read_only": True},
        {"type": "volume", "source": "myvol", "target": "/v"},
    ]
    env = _make_env(tmp_path, mounts=mounts)
    asyncio.run(env.start(force_build=False))

    run_argv = captured_argv[2]
    assert "/host/data:/data" in run_argv
    assert "/host/logs:/logs:ro" in run_argv
    assert "myvol:/v" not in run_argv


def test_start_runs_pre_mount_chmod_for_writable_targets(tmp_path, captured_argv):
    mounts = [{"type": "bind", "source": "/host/logs", "target": "/logs/agent"}]
    env = _make_env(tmp_path, mounts=mounts)
    asyncio.run(env.start(force_build=False))

    exec_argv = captured_argv[3]
    assert "exec" in exec_argv
    assert "--user" in exec_argv and "0" in exec_argv
    cmd_str = " ".join(exec_argv)
    assert "mkdir -p" in cmd_str
    assert "chmod 0777" in cmd_str
    assert "/logs/agent" in cmd_str


def test_start_force_build_runs_podman_build(tmp_path, captured_argv):
    env = _make_env(tmp_path, docker_image=None)
    asyncio.run(env.start(force_build=True))

    build_argv = captured_argv[1]
    assert "build" in build_argv
    assert "-t" in build_argv
    assert "-f" in build_argv


# exec()


def test_exec_argv_with_cwd_env_and_user(tmp_path, captured_argv):
    env = _make_env(tmp_path)
    asyncio.run(
        env.exec(
            "python /data/run_evaluation_curated.py --out /logs/reward.json",
            cwd="/workspace",
            env={"VERIFIER_URL": "http://127.0.0.1:5775", "AGENT_ID": "agent-1"},
            timeout_sec=600,
            user="root",
        )
    )

    argv = captured_argv[0]
    assert argv[0] == "podman"
    assert "exec" in argv
    w_idx = argv.index("-w")
    assert argv[w_idx + 1] == "/workspace"
    e_values = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
    assert "VERIFIER_URL=http://127.0.0.1:5775" in e_values
    assert "AGENT_ID=agent-1" in e_values
    u_idx = argv.index("--user")
    assert argv[u_idx + 1] == "root"
    assert argv[-3] == "bash"
    assert argv[-2] == "-c"
    assert "python /data/run_evaluation_curated.py" in argv[-1]


# upload / download — podman cp argv


def test_upload_file_argv(tmp_path, captured_argv):
    env = _make_env(tmp_path, session_id="ofg-up")
    src = tmp_path / "local.txt"
    src.write_text("hello")
    asyncio.run(env.upload_file(src, "/data/local.txt"))

    argv = captured_argv[0]
    assert "cp" in argv
    assert str(src) in argv
    assert any(a.endswith(":/data/local.txt") for a in argv)


def test_download_file_argv(tmp_path, captured_argv):
    env = _make_env(tmp_path, session_id="ofg-dn")
    target = tmp_path / "downloaded.txt"
    asyncio.run(env.download_file("/data/result.txt", target))

    argv = captured_argv[0]
    assert "cp" in argv
    assert any(a.endswith(":/data/result.txt") for a in argv)
    assert str(target) in argv


def test_upload_dir_creates_target_then_copies(tmp_path, captured_argv):
    env = _make_env(tmp_path)
    src = tmp_path / "srcdir"
    src.mkdir()
    asyncio.run(env.upload_dir(src, "/data/dst"))

    assert len(captured_argv) >= 2
    mkdir_argv = captured_argv[0]
    cp_argv = captured_argv[1]
    assert "exec" in mkdir_argv
    assert "mkdir -p" in mkdir_argv[-1]
    assert "/data/dst" in mkdir_argv[-1]
    assert "cp" in cp_argv


# stop()


def test_stop_without_delete(tmp_path, captured_argv):
    env = _make_env(tmp_path)
    asyncio.run(env.start(force_build=False))
    captured_argv.clear()
    asyncio.run(env.stop(delete=False))

    assert len(captured_argv) == 1
    stop_argv = captured_argv[0]
    assert "stop" in stop_argv
    assert "--time" in stop_argv
    # `rm` should not be in this argv (no delete).
    assert "rm" not in stop_argv


def test_stop_with_delete(tmp_path, captured_argv):
    env = _make_env(tmp_path)
    asyncio.run(env.start(force_build=False))
    captured_argv.clear()
    asyncio.run(env.stop(delete=True))

    assert len(captured_argv) == 2
    assert "stop" in captured_argv[0]
    rm_argv = captured_argv[1]
    assert "rm" in rm_argv
    assert "--force" in rm_argv


def test_stop_is_idempotent_before_start(tmp_path, captured_argv):
    env = _make_env(tmp_path)
    asyncio.run(env.stop(delete=True))
    assert captured_argv == []
