#!/usr/bin/env python
"""Remove verifier runs whose referenced trial directories no longer exist.

Current ledgers are matched by ``trial_dir``; older ledgers are matched through
trial sidecars. The default is a dry run. ``--apply`` requires confirmation
unless ``--yes`` is supplied.

Run from the repo root::

    python scripts/maintenance/cleanup_stale_verifier_runs.py            # dry-run (default)
    python scripts/maintenance/cleanup_stale_verifier_runs.py --apply    # delete after Y/N prompt per dir
    python scripts/maintenance/cleanup_stale_verifier_runs.py --apply --yes   # batch delete
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Optional


_DEFAULT_VERIFIER_ROOT = Path("data/run_output/verifier")
_DEFAULT_EXAMPLES_ROOT = Path("data/run_output/examples")
_DB_GLOB = "*/runs/*/predictions.db"
_SIDECAR_NAME = "deferred_session.json"


# Local read-only SQLite helpers keep the maintenance script self-contained.


def _open_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _has_column(conn: sqlite3.Connection, col: str) -> bool:
    rows = conn.execute("PRAGMA table_info(predictions)").fetchall()
    return any(r["name"] == col for r in rows)


def _ledger_session_trial_dirs(conn: sqlite3.Connection) -> list[Optional[str]]:
    """Return one trial_dir per session_id. May contain None for legacy
    rows whose session was submitted before the trial_dir column existed."""
    if not _has_column(conn, "trial_dir"):
        sids = conn.execute(
            "SELECT DISTINCT session_id FROM predictions"
        ).fetchall()
        return [None] * len(sids)
    rows = conn.execute(
        "SELECT session_id, MAX(trial_dir) AS td "
        "FROM predictions GROUP BY session_id"
    ).fetchall()
    return [r["td"] for r in rows]


# Sidecar-scan fallback (legacy DBs without trial_dir column)


def _ledger_paths_referenced_by_sidecars(examples_root: Path) -> set[str]:
    """Set of absolute, resolved ledger_path strings any sidecar still
    points at. A ledger whose path isn't in this set is unreferenced."""
    out: set[str] = set()
    if not examples_root.exists():
        return out
    for sidecar in examples_root.rglob(_SIDECAR_NAME):
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        lp = data.get("ledger_path")
        if not lp:
            continue
        try:
            out.add(str(Path(lp).resolve()))
        except OSError:
            continue
    return out


# Orphan classification


def _classify_orphan(
    db_path: Path,
    examples_root: Path,
    referenced_ledgers: set[str],
) -> tuple[bool, str]:
    """Return ``(is_orphan, reason)``.

    Decision tree:

    1. DB has trial_dir column → orphan iff all stamped trial_dirs are
       missing from disk. ``None`` values are treated as missing — if
       a session never stamped a trial_dir, we fall back to step 2.
    2. DB has no column OR no stamped trial_dirs → orphan iff no
       deferred_session.json under examples_root references this DB.

    Empty ledger (no rows at all) is reported as orphan with that reason.
    """
    try:
        conn = _open_ro(db_path)
    except sqlite3.Error as exc:
        return False, f"unreadable: {exc}"
    try:
        if not conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='predictions'"
        ).fetchone():
            return True, "not a ledger DB"
        n_rows = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions"
        ).fetchone()["c"]
        if n_rows == 0:
            return True, "empty ledger"
        session_dirs = _ledger_session_trial_dirs(conn)
    finally:
        conn.close()

    stamped = [d for d in session_dirs if d]
    if stamped:
        missing = [d for d in stamped if not Path(d).exists()]
        if len(missing) == len(stamped):
            return True, f"all {len(stamped)} stamped trial_dir(s) deleted"
        return False, f"{len(stamped) - len(missing)}/{len(stamped)} trial_dir(s) still alive"

    # Legacy fallback — no stamped trial_dir on any session.
    key = str(db_path.resolve())
    if key in referenced_ledgers:
        return False, "referenced by examples/<sidecar>"
    return True, "no stamped trial_dir AND no sidecar references this DB"


# Main


def _prompt_yes_no(question: str, *, default_yes: bool = True) -> bool:
    """Single-shot Y/N — same defensive shape as inspect_ledger."""
    suffix = " [Y/n] " if default_yes else " [y/N] "
    sys.stdout.flush()
    try:
        answer = input(question + suffix).strip().lower()
    except EOFError:
        return default_yes
    if answer == "":
        return default_yes
    if answer in ("y", "yes"):
        return True
    if answer in ("n", "no"):
        return False
    print(f"  (unrecognised answer {answer!r}, treating as no)")
    return False


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="cleanup_stale_verifier_runs",
        description=(
            "Remove verifier run dirs whose trial dir under examples/ "
            "has been deleted. Dry-run by default — pass --apply to "
            "actually delete."
        ),
    )
    p.add_argument(
        "--root", type=Path, default=_DEFAULT_VERIFIER_ROOT,
        help=f"Verifier root to scan (default: {_DEFAULT_VERIFIER_ROOT}).",
    )
    p.add_argument(
        "--examples-root", type=Path, default=_DEFAULT_EXAMPLES_ROOT,
        help=f"Examples root for sidecar fallback (default: {_DEFAULT_EXAMPLES_ROOT}).",
    )
    p.add_argument(
        "--apply", action="store_true",
        help="Actually delete the orphan run dirs (default is dry-run).",
    )
    p.add_argument(
        "--yes", action="store_true",
        help="With --apply: skip the per-dir Y/N prompt.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    root = args.root.resolve()
    examples_root = args.examples_root.resolve()

    if not root.exists():
        print(f"[cleanup] no verifier root at {root}", file=sys.stderr)
        return 1

    db_paths = sorted(root.glob(_DB_GLOB))
    if not db_paths:
        print(f"[cleanup] no ledger DBs found under {root}")
        return 0

    referenced = _ledger_paths_referenced_by_sidecars(examples_root)

    orphans: list[tuple[Path, str]] = []
    alive: list[tuple[Path, str]] = []
    for db in db_paths:
        is_orphan, reason = _classify_orphan(db, examples_root, referenced)
        (orphans if is_orphan else alive).append((db, reason))

    print(f"Scanned {len(db_paths)} ledger DB(s) under {root}:")
    print(f"  alive:   {len(alive)}")
    print(f"  orphans: {len(orphans)}")
    if alive:
        print("\nAlive (kept):")
        for db, reason in alive:
            print(f"  {db.parent}  ({reason})")
    if not orphans:
        print("\nNothing to clean up.")
        return 0

    print("\nOrphans:")
    for db, reason in orphans:
        # The unit we delete is the run dir (parent of predictions.db).
        run_dir = db.parent
        # Try to report total size for the user's benefit.
        try:
            size = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
            size_str = f"{size / 1024:.1f} KiB"
        except OSError:
            size_str = "?"
        print(f"  {run_dir}  ({reason}, {size_str})")

    if not args.apply:
        print(
            f"\nDry-run — no changes made. Re-run with --apply to delete "
            f"({len(orphans)} dir(s))."
        )
        return 0

    # --apply path. Confirm per-dir unless --yes.
    deleted = 0
    for db, reason in orphans:
        run_dir = db.parent
        question = f"Delete {run_dir}? ({reason})"
        if not args.yes and not _prompt_yes_no(question, default_yes=True):
            print(f"  skipped {run_dir}")
            continue
        try:
            shutil.rmtree(run_dir)
            deleted += 1
            print(f"  deleted {run_dir}")
        except OSError as exc:
            print(f"  FAILED to delete {run_dir}: {exc}", file=sys.stderr)

    print(f"\nDeleted {deleted}/{len(orphans)} orphan run dir(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
