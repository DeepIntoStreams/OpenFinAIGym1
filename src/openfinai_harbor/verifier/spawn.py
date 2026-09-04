"""Lazily maintain one verifier process per task.

A health fast path and double-checked file lock coordinate concurrent agents.
The winner starts a detached process, waits for health, and atomically records
its endpoint. Failures terminate the child and write ``spawn_failed.json``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from filelock import FileLock, Timeout as FileLockTimeout

from openfinai_harbor.verifier import auth, registry


_HEALTHZ_FAST_PATH_TIMEOUT_SEC = 2.0
_READY_POLL_INTERVAL_SEC = 0.2
# Cold-start budget covers imports + on_startup (CSV reads + optional live
# Alpaca/Binance prefetch) + uvicorn bind. Windows cold imports can exceed
# 30s; 60s is headroom. Warm spawns exit as soon as /healthz returns 200.
_READY_POLL_TOTAL_SEC = 60.0
_LOCK_ACQUIRE_TIMEOUT_SEC = 30.0


@dataclasses.dataclass(frozen=True)
class VerifierEndpoint:
    """Connection details for a running verifier."""

    task_id: str
    url: str
    token: str
    pid: int
    state_dir: Path

    def auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


class VerifierSpawnError(RuntimeError):
    """Raised when spawn coordination fails (lock timeout, child dies, etc.)."""


def task_id_from_task_dir(task_dir: Path) -> str:
    """Resolve task_id from an installed task layout.

    Convention: ``tasks/generated/<scope>/<task_id>/``. ``task_dir.name``
    is the canonical task_id. ``task.toml`` carries
    ``[metadata] name = "<scope>/<task_id>"`` but is optional.
    """
    return task_dir.name


def _probe_existing(state_dir: Path) -> Optional[VerifierEndpoint]:
    """Try to attach to an already-running verifier without taking the lock.

    Returns the endpoint iff the url-file exists, the recorded process is
    alive, AND ``/healthz`` returns 200. Otherwise returns None and the
    caller falls through to the spawn path.
    """
    info = registry.read_url_file(state_dir)
    if info is None:
        return None
    pid = info.get("pid")
    if isinstance(pid, int) and not registry.pid_is_alive(pid):
        # Stale url-file: process gone but file orphaned. New spawn will
        # overwrite atomically.
        return None
    url = info.get("url")
    token = info.get("token")
    task_id = info.get("task_id")
    if not (isinstance(url, str) and isinstance(token, str) and isinstance(task_id, str)):
        return None
    try:
        resp = requests.get(
            url.rstrip("/") + "/healthz", timeout=_HEALTHZ_FAST_PATH_TIMEOUT_SEC
        )
        if resp.status_code != 200:
            return None
    except (requests.ConnectionError, requests.Timeout):
        return None
    return VerifierEndpoint(
        task_id=task_id,
        url=url,
        token=token,
        pid=int(pid) if isinstance(pid, int) else 0,
        state_dir=state_dir,
    )


def _pick_unused_port(bind: str = "127.0.0.1") -> int:
    """Bind to port 0, read the OS-assigned port, close.

    There is a tiny race window between this function returning and the
    verifier subprocess re-binding the same port. If another process
    grabs it in between, the verifier will fail to start and we surface
    a clean spawn error.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((bind, 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _spawn_child(
    *,
    task_dir: Path,
    port: int,
    bind: str,
    token: str,
    despawn_grace_min: float,
    state_dir: Path,
    log_path: Path,
    max_requests_per_minute: int | None = None,
    extra_python_args: Optional[list[str]] = None,
) -> subprocess.Popen[bytes]:
    """Detach + spawn the verifier subprocess.

    On POSIX: ``start_new_session=True`` puts the child in its own
    session so SIGINT in the spawner does not propagate.
    On Windows: ``CREATE_NEW_PROCESS_GROUP`` achieves the same.
    """
    # NB: `--token=<value>` (joined with `=`) — secrets.token_urlsafe() can
    # produce strings starting with `-`, which argparse would otherwise
    # mistake for the next option and reject as "expected one argument".
    cmd = [
        sys.executable,
        "-m",
        "openfinai_harbor.verifier",
        "--task-dir",
        str(task_dir),
        "--port",
        str(port),
        "--bind",
        bind,
        f"--token={token}",
        "--despawn-grace-min",
        f"{despawn_grace_min}",
        "--state-dir",
        str(state_dir),
    ]
    if max_requests_per_minute is not None:
        # RL pipelines burst hundreds of /submit per second; default 60/min
        # (verifier/__main__.py) throttles rollouts. Default is opt-in.
        cmd += ["--max-requests-per-minute", str(max_requests_per_minute)]
    if extra_python_args:
        cmd[1:1] = list(extra_python_args)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Unbuffered so each log line lands promptly.
    log_fp = log_path.open("ab", buffering=0)
    creationflags = 0
    start_new_session = False
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        start_new_session = True

    return subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=log_fp,
        stdin=subprocess.DEVNULL,
        cwd=str(Path.cwd()),
        env=_subprocess_env(),
        creationflags=creationflags,
        start_new_session=start_new_session,
        close_fds=True,
    )


def _subprocess_env() -> dict[str, str]:
    """Build env for the verifier child.

    Propagates the current process's ``sys.path`` entries (filtered to
    actual filesystem paths) into ``PYTHONPATH`` so the child can resolve
    ``openfinai_harbor.verifier`` whether the parent installed the
    package via ``pip install -e .`` / ``poetry install`` or set its
    own ``PYTHONPATH``. Never blanks an existing PYTHONPATH — we
    prepend our entries.
    """
    env = os.environ.copy()
    parent_paths = [p for p in sys.path if p and Path(p).exists()]
    existing = env.get("PYTHONPATH", "")
    parts = parent_paths + ([existing] if existing else [])
    if parts:
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _wait_for_ready(
    *,
    bind: str,
    port: int,
    token: str,
    deadline: float,
    child: subprocess.Popen[bytes],
) -> bool:
    """Poll /healthz until 200 (or child dies / timeout).

    Returns True on success, False on failure. Caller decides whether to
    raise based on the failure mode (timeout vs child dead).
    """
    target = f"http://{_loopback_for_bind(bind)}:{port}/healthz"
    while time.time() < deadline:
        if child.poll() is not None:
            return False  # child exited before becoming ready
        try:
            resp = requests.get(target, timeout=0.5)
            if resp.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(_READY_POLL_INTERVAL_SEC)
    return False


def _loopback_for_bind(bind: str) -> str:
    """Pick the address a host-side caller should use given the bind interface.

    Bind ``0.0.0.0`` is reachable via 127.0.0.1 from the same host. Bind
    127.0.0.1 is the same address. For docker-network binds this
    function would need extending — out of scope for the host-process
    spawn coordinator.
    """
    if bind == "0.0.0.0":
        return "127.0.0.1"
    return bind


def get_or_spawn(
    task_dir: Path,
    *,
    despawn_grace_min: float = 10.0,
    bind: str = "127.0.0.1",
    run_output_root: Optional[Path] = None,
    max_requests_per_minute: int | None = None,
) -> VerifierEndpoint:
    """Return a live ``VerifierEndpoint`` for ``task_dir`` (spawn if needed).

    Idempotent across concurrent callers via filelock. On spawn failure,
    raises ``VerifierSpawnError`` with diagnostic context.

    ``max_requests_per_minute`` overrides the verifier's per-bearer-token
    rate limit (default 60/min). RL training generates many submits per
    second per task — set this high (e.g. 100000) when prewarming
    verifiers for an RL run. ``None`` keeps the verifier's default.
    """
    task_dir = task_dir.resolve()
    if not task_dir.exists():
        raise VerifierSpawnError(f"task_dir does not exist: {task_dir}")
    task_id = task_id_from_task_dir(task_dir)
    state_dir = registry.state_dir_for_task(
        task_id=task_id, run_output_root=run_output_root
    )
    state_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fast path.
    existing = _probe_existing(state_dir)
    if existing is not None:
        return existing

    # 2. Acquire lock for spawn.
    lock = FileLock(str(registry.lock_file(state_dir)))
    try:
        lock.acquire(timeout=_LOCK_ACQUIRE_TIMEOUT_SEC)
    except FileLockTimeout as exc:
        raise VerifierSpawnError(
            f"could not acquire spawn lock for task_id={task_id} within "
            f"{_LOCK_ACQUIRE_TIMEOUT_SEC}s; another spawn may be stuck. "
            f"Check {registry.lock_file(state_dir)}"
        ) from exc

    try:
        # 3. Re-check after acquire (another caller may have spawned).
        existing = _probe_existing(state_dir)
        if existing is not None:
            return existing

        # 4. Pick port + token.
        port = _pick_unused_port(bind=_loopback_for_bind(bind))
        token = auth.generate_token()

        # 5. Prepare run-dir + log path.
        run_dir = registry.new_run_dir(state_dir)
        log_path = registry.verifier_log_path(run_dir)

        # 6. Spawn child.
        child = _spawn_child(
            task_dir=task_dir,
            port=port,
            bind=bind,
            token=token,
            despawn_grace_min=despawn_grace_min,
            state_dir=state_dir,
            log_path=log_path,
            max_requests_per_minute=max_requests_per_minute,
        )

        # 7. Wait for ready.
        deadline = time.time() + _READY_POLL_TOTAL_SEC
        ok = _wait_for_ready(
            bind=bind, port=port, token=token, deadline=deadline, child=child
        )
        if not ok:
            failure_reason = _diagnose_spawn_failure(child=child, log_path=log_path)
            try:
                child.kill()
            except Exception:
                pass
            (run_dir / "spawn_failed.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "task_dir": str(task_dir),
                        "port": port,
                        "bind": bind,
                        "reason": failure_reason,
                        "log_path": str(log_path),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise VerifierSpawnError(
                f"verifier did not become ready within "
                f"{_READY_POLL_TOTAL_SEC}s for task_id={task_id}; "
                f"reason={failure_reason}; log={log_path}"
            )

        # 8. Atomically write url-file + pid file.
        host_url = f"http://{_loopback_for_bind(bind)}:{port}"
        registry.write_url_file_atomic(
            state_dir,
            url=host_url,
            token=token,
            task_id=task_id,
            pid=child.pid,
        )
        registry.write_pid_file(state_dir, child.pid)

        return VerifierEndpoint(
            task_id=task_id,
            url=host_url,
            token=token,
            pid=child.pid,
            state_dir=state_dir,
        )
    finally:
        lock.release()


def _diagnose_spawn_failure(
    *, child: subprocess.Popen[bytes], log_path: Path
) -> str:
    """Best-effort failure diagnosis for the spawn_failed.json + error message."""
    rc = child.poll()
    if rc is not None and rc != 0:
        tail = ""
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            tail = text[-2000:]
        except OSError:
            pass
        return f"child exited rc={rc}; tail of log:\n{tail}"
    if rc is None:
        return "child still running but /healthz not reachable within timeout"
    return "child exited rc=0 before becoming ready (unusual)"


def request_despawn(endpoint: VerifierEndpoint, *, agent_id: str) -> None:
    """POST /deregister so a grace=0 verifier shuts down promptly.

    Best-effort: failures are silent (the lifecycle thread will catch up
    on its next tick anyway).
    """
    try:
        requests.post(
            endpoint.url.rstrip("/") + "/deregister",
            json={"agent_id": agent_id},
            headers=endpoint.auth_header(),
            timeout=2.0,
        )
    except Exception:
        pass
