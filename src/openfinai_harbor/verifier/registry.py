"""Per-task verifier state-directory layout helpers.

State-dir contract::

    data/run_output/verifier/<task_id>/
    ├── verifier.lock          (filelock for spawn coordination)
    ├── verifier.url           (JSON: url + token + pid + started_at_utc)
    ├── verifier.pid           (textual PID for diagnostics)
    └── runs/<utc_ts>/
        ├── verifier.log       (uvicorn access + app log)
        ├── submissions.jsonl  (audit, one line per submission)
        └── crashed.json       (only on crash; traceback + last submission)

The url-file is the canonical point of contact: any agent that knows the
state-dir for a task can read it to discover the verifier endpoint and
auth token. Mode 0o600 on POSIX. (Windows: relies on directory ACLs;
documented caveat.)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_VERIFIER_LOG_FILENAME = "verifier.log"
_VERIFIER_URL_FILENAME = "verifier.url"
_VERIFIER_PID_FILENAME = "verifier.pid"
_VERIFIER_LOCK_FILENAME = "verifier.lock"
_VERIFIER_SUBMISSIONS_FILENAME = "submissions.jsonl"
_VERIFIER_CRASHED_FILENAME = "crashed.json"


def utc_now_iso() -> str:
    """Return ``YYYY-MM-DDTHH:MM:SSZ`` for url-file / log timestamps."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now_compact() -> str:
    """Return ``YYYYMMDDTHHMMSSZ`` for run-dir naming."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def state_dir_for_task(
    *, task_id: str, run_output_root: Optional[Path] = None
) -> Path:
    """Return the state directory path for a given task_id.

    ``run_output_root`` defaults to ``<repo_root>/data/run_output``
    (computed by walking up from this file's location). Tests pass an
    explicit root to avoid touching the real output tree.
    """
    if run_output_root is None:
        run_output_root = _default_run_output_root()
    return run_output_root / "verifier" / task_id


def _default_run_output_root() -> Path:
    """Walk up from this module to find the repo root + ``data/run_output``."""
    here = Path(__file__).resolve()
    return here.parents[3] / "data" / "run_output"


def url_file(state_dir: Path) -> Path:
    return state_dir / _VERIFIER_URL_FILENAME


def lock_file(state_dir: Path) -> Path:
    return state_dir / _VERIFIER_LOCK_FILENAME


def pid_file(state_dir: Path) -> Path:
    return state_dir / _VERIFIER_PID_FILENAME


def runs_dir(state_dir: Path) -> Path:
    return state_dir / "runs"


def new_run_dir(state_dir: Path) -> Path:
    """Create + return a fresh ``runs/<utc_ts>/`` directory for this run."""
    rd = runs_dir(state_dir) / utc_now_compact()
    rd.mkdir(parents=True, exist_ok=True)
    return rd


def submissions_log_path(run_dir: Path) -> Path:
    return run_dir / _VERIFIER_SUBMISSIONS_FILENAME


def crashed_marker_path(run_dir: Path) -> Path:
    return run_dir / _VERIFIER_CRASHED_FILENAME


def verifier_log_path(run_dir: Path) -> Path:
    return run_dir / _VERIFIER_LOG_FILENAME


def write_url_file_atomic(
    state_dir: Path,
    *,
    url: str,
    token: str,
    task_id: str,
    pid: int,
) -> Path:
    """Atomically write the url-file (tmp + os.replace)."""
    state_dir.mkdir(parents=True, exist_ok=True)
    target = url_file(state_dir)
    tmp = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "url": url,
        "token": token,
        "task_id": task_id,
        "pid": pid,
        "started_at_utc": utc_now_iso(),
    }
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if sys.platform != "win32":
        # Restrict read access to the spawning user where the FS supports it.
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
    os.replace(tmp, target)
    return target


def read_url_file(state_dir: Path) -> Optional[Dict[str, Any]]:
    """Read and parse the url-file. Returns None if missing or unparseable."""
    target = url_file(state_dir)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def remove_url_file(state_dir: Path) -> None:
    """Best-effort delete of the url-file (used at clean shutdown)."""
    target = url_file(state_dir)
    try:
        target.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def write_pid_file(state_dir: Path, pid: int) -> None:
    pid_file(state_dir).write_text(str(pid) + "\n", encoding="utf-8")


def read_pid_file(state_dir: Path) -> Optional[int]:
    target = pid_file(state_dir)
    if not target.exists():
        return None
    try:
        return int(target.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def pid_is_alive(pid: int) -> bool:
    """Cross-platform process-liveness check.

    POSIX: ``os.kill(pid, 0)`` raises if the process is gone.
    Windows: ``OpenProcess``-style check via ``ctypes``; PIDs can be
    recycled but this is good enough to detect orphan url-files.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _pid_is_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # We can't signal it but it does exist.
        return True
    except OSError:
        return False
    # os.kill(pid, 0) succeeds for zombie processes too; check for zombie state.
    try:
        stat_path = f"/proc/{pid}/status"
        with open(stat_path) as f:
            for line in f:
                if line.startswith("State:") and "Z" in line:
                    return False
    except OSError:
        pass
    return True


def _pid_is_alive_windows(pid: int) -> bool:
    import ctypes
    import ctypes.wintypes as wt

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = wt.DWORD()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        if not ok:
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)
