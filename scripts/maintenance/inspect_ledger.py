#!/usr/bin/env python
"""Inspect deferred-prediction ledgers and optionally resolve ready rows.

Inspection uses read-only database connections. Interactive runs offer to
resolve ready trials; ``--no-resolve`` disables mutation and ``--yes`` enables
non-interactive resolution.

Run from the repo root::

    python scripts/maintenance/inspect_ledger.py                    # summary; auto-prompts if ready rows in TTY
    python scripts/maintenance/inspect_ledger.py --ready            # rows whose horizon has passed
    python scripts/maintenance/inspect_ledger.py --failed -v        # failed rows + full error_message
    python scripts/maintenance/inspect_ledger.py --trial-dir <dir>  # one trial, all rows
    python scripts/maintenance/inspect_ledger.py --no-resolve       # never prompt (read-only)
    python scripts/maintenance/inspect_ledger.py --yes              # batch resolve without prompting
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


# Path layout (matches realtime_forecasting.py + realtime_polymarket.py).
# DBs live at data/run_output/verifier/<task_id>/runs/<utc_ts>/predictions.db
_DEFAULT_VERIFIER_ROOT = Path("data/run_output/verifier")
_DEFAULT_EXAMPLES_ROOT = Path("data/run_output/examples")
_DB_GLOB = "*/runs/*/predictions.db"
_DB_NAME = "predictions.db"
_SIDECAR_NAME = "deferred_session.json"


# Discovery


def _discover_ledgers(root: Path) -> list[Path]:
    """Find every per-trial ledger under *root*. Sorted oldest-first."""
    if not root.exists():
        return []
    return sorted(root.glob(_DB_GLOB))


def _path_facets(db_path: Path) -> dict[str, str]:
    """Derive (task, trial) labels from the ledger's path.

    Layout: ``.../verifier/<task>/runs/<trial>/predictions.db``.
    Falls back to ``"?"`` when the path doesn't match the expected layout.
    """
    parts = db_path.parts
    task = trial = "?"
    try:
        v_idx = parts.index("verifier")
        task = parts[v_idx + 1]
        if parts[v_idx + 2] == "runs":
            trial = parts[v_idx + 3]
    except (ValueError, IndexError):
        pass
    return {"task": task, "trial": trial}


def _abbrev_task(task: str) -> str:
    """Compact task label for the row view (drop the `realtime_` prefix)."""
    if task.startswith("realtime_"):
        return task[len("realtime_"):]
    return task


def _build_trial_dir_index(examples_root: Path) -> dict[str, Path]:
    """Scan ``examples_root`` for ``verifier/deferred_session.json`` sidecars
    and build a ``{ledger_path → trial_dir}`` reverse map.

    Legacy fallback for ledgers whose rows pre-date the ``trial_dir``
    column (DBs created before 2026-05-14). New submissions stamp the
    trial path directly onto each row — see ``_trial_dir_from_ledger``.
    """
    index: dict[str, Path] = {}
    if not examples_root.exists():
        return index
    for sidecar in examples_root.rglob(_SIDECAR_NAME):
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ledger_path = data.get("ledger_path")
        if not ledger_path:
            continue
        # The trial_dir is the parent of the `verifier/` directory that
        # contains the sidecar — i.e. sidecar.parent.parent.
        try:
            trial_dir = sidecar.parent.parent
        except IndexError:
            continue
        # Normalize ledger_path for cross-platform key consistency.
        index[str(Path(ledger_path).resolve())] = trial_dir.resolve()
    return index


def _trial_dir_from_ledger(conn: sqlite3.Connection) -> Optional[Path]:
    """Read the ledger row's ``trial_dir`` column.

    Stamped by the curated handler at submission time when the agent
    forwarded ``HARBOR_TRIAL_DIR``. Returns the FIRST non-null value —
    one ledger DB normally has one trial_dir even if multiple sessions
    share it. Returns ``None`` when the column doesn't exist (legacy
    DB) or when every row has trial_dir=NULL.
    """
    if not _has_column(conn, "trial_dir"):
        return None
    row = conn.execute(
        "SELECT trial_dir FROM predictions "
        "WHERE trial_dir IS NOT NULL LIMIT 1"
    ).fetchone()
    if row is None or not row["trial_dir"]:
        return None
    return Path(row["trial_dir"])


def _resolve_trial_dir_to_ledger(trial_dir: Path) -> Optional[tuple[Path, str]]:
    """Read deferred_session.json under *trial_dir* and return (db, session_id)."""
    candidates = [
        trial_dir / "verifier" / _SIDECAR_NAME,
        trial_dir / _SIDECAR_NAME,
    ]
    for c in candidates:
        if c.exists():
            break
    else:
        matches = list(trial_dir.rglob(_SIDECAR_NAME))
        if not matches:
            return None
        c = matches[0]
    try:
        sidecar = json.loads(c.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    db = sidecar.get("ledger_path")
    sid = sidecar.get("session_id")
    if not db or not sid:
        return None
    return Path(db), sid


# SQLite (read-only)


def _open_ro(db_path: Path) -> sqlite3.Connection:
    """Open *db_path* in read-only mode (no migration, no writes)."""
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _is_ledger(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
    ).fetchone()
    return row is not None


def _has_column(conn: sqlite3.Connection, col: str) -> bool:
    """SQLite PRAGMA column lookup. Used to gate optional reads on old DBs
    (e.g. ``predicted_return``/``confidence`` columns dropped from the
    schema but still present on legacy DBs)."""
    rows = conn.execute("PRAGMA table_info(predictions)").fetchall()
    return any(r["name"] == col for r in rows)


def _select_rows(
    conn: sqlite3.Connection,
    *,
    statuses: Optional[set[str]],
    session_id: Optional[str],
    only_overdue: bool,
    only_waiting: bool,
    now_iso: str,
    limit: int,
) -> list[sqlite3.Row]:
    where: list[str] = []
    params: list[Any] = []
    if statuses:
        placeholders = ",".join("?" * len(statuses))
        where.append(f"status IN ({placeholders})")
        params.extend(sorted(statuses))
    if session_id:
        where.append("session_id = ?")
        params.append(session_id)
    if only_overdue:
        # --ready: pending rows whose resolve_at has elapsed. Constrain
        # status here so --ready stacks correctly with other filters.
        where.append("status = 'pending' AND resolve_at <= ?")
        params.append(now_iso)
    if only_waiting:
        # --waiting: pending rows whose horizon is still in the future.
        where.append("status = 'pending' AND resolve_at > ?")
        params.append(now_iso)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    sql = (
        f"SELECT id, session_id, provider, symbol, status, resolve_at, "
        f"submitted_at, direction, predicted_price, entry_price, "
        f"exit_price, actual_return, snapshot, error_message "
        f"FROM predictions{clause} ORDER BY resolve_at LIMIT ?"
    )
    params.append(limit)
    return conn.execute(sql, params).fetchall()


def _summary_counts(conn: sqlite3.Connection, *, now_iso: str) -> dict[str, int]:
    """Per-status counts, with `pending` split into `waiting` vs `ready`."""
    rows = conn.execute(
        "SELECT status, COUNT(*) AS cnt FROM predictions GROUP BY status"
    ).fetchall()
    raw = {r["status"]: r["cnt"] for r in rows}
    pending = int(raw.get("pending", 0))
    if pending > 0:
        ready_row = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions "
            "WHERE status='pending' AND resolve_at <= ?",
            (now_iso,),
        ).fetchone()
        ready = int(ready_row["c"]) if ready_row else 0
    else:
        ready = 0
    waiting = pending - ready
    # Collapse resolved + scored into a single "done" bucket — in
    # practice the resolver transitions both in a single run_once(), so
    # operators don't need to distinguish them at a glance.
    done = int(raw.get("resolved", 0)) + int(raw.get("scored", 0))
    return {
        "total": int(sum(raw.values())),
        "waiting": waiting,
        "ready": ready,
        "done": done,
        "failed": int(raw.get("failed", 0)),
    }


def _next_resolve_at(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute(
        "SELECT MIN(resolve_at) AS m FROM predictions WHERE status='pending'"
    ).fetchone()
    return row["m"] if row and row["m"] else None


# Rendering


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print(" (no rows)")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def _format_resolve_at(resolve_at: str) -> str:
    """Compact absolute ISO timestamp at second precision.

    Strips microseconds and normalises a ``+00:00`` UTC offset to ``Z``
    so the column reads e.g. ``2026-05-12T01:28:54Z``. Falls back to the
    raw value when the timestamp doesn't parse — better to display
    something than to silently drop it.
    """
    if not resolve_at:
        return ""
    try:
        dt = datetime.fromisoformat(resolve_at)
    except ValueError:
        return resolve_at
    dt = dt.replace(microsecond=0)
    if dt.utcoffset() == timedelta(0):
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.isoformat()


def _truncate(s: Optional[str], n: int) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").replace("\r", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _format_number(v: Any) -> str:
    """Compact numeric column. NULL → ''; probabilities (0-1) get 2dp;
    larger magnitudes get 2dp; very small (<1e-4) get scientific."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == 0:
        return "0"
    af = abs(f)
    if af < 1e-4:
        return f"{f:.2e}"
    if af < 1.0:
        return f"{f:.3f}"
    if af < 1000.0:
        return f"{f:.2f}"
    return f"{f:.2f}"


def _row_label(row: sqlite3.Row, *, verbose: bool) -> str:
    """Return the symbol-or-question label for a row.

    Polymarket rows: prefer ``snapshot.question`` (persisted at submit
    time post 2026-05-13). Legacy polymarket rows pre-snapshot-change
    fall back to truncated condition_id. Other families: just symbol.
    """
    symbol = row["symbol"] or ""
    snap_raw = row["snapshot"]
    question: Optional[str] = None
    if snap_raw:
        try:
            snap = json.loads(snap_raw)
            if isinstance(snap, dict):
                q = snap.get("question")
                if isinstance(q, str) and q.strip():
                    question = q.strip()
        except (ValueError, TypeError):
            pass
    if question is not None:
        return question if verbose else _truncate(question, 50)
    return symbol if verbose else _truncate(symbol, 24)


def _print_summary_view(
    discoveries: list[tuple[Path, sqlite3.Connection, dict[str, int]]],
    now: datetime,
) -> int:
    """Render the cross-trial summary. Returns the total ready-count
    so callers can decide whether to show the --resolve hint."""
    headers = [
        "task", "trial", "total", "waiting", "ready", "done",
        "failed", "next_resolve_at",
    ]
    rows: list[list[str]] = []
    total_ready = 0
    for db, conn, counts in discoveries:
        facets = _path_facets(db)
        next_at = _format_resolve_at(_next_resolve_at(conn) or "")
        total_ready += counts.get("ready", 0)
        rows.append([
            facets["task"], facets["trial"],
            str(counts.get("total", 0)),
            str(counts.get("waiting", 0)),
            str(counts.get("ready", 0)),
            str(counts.get("done", 0)),
            str(counts.get("failed", 0)),
            next_at,
        ])
    _print_table(headers, rows)
    return total_ready


def _print_row_view(
    discoveries: list[tuple[Path, sqlite3.Connection, dict[str, int]]],
    now: datetime,
    *,
    statuses: Optional[set[str]],
    session_id: Optional[str],
    only_overdue: bool,
    only_waiting: bool,
    limit_per_db: int,
    verbose: bool,
) -> None:
    headers = ["task", "pred_id", "symbol-or-question",
               "predicted", "entry", "submitted_at", "resolve_at",
               "status", "error"]
    if verbose:
        headers.insert(2, "trial")
    out_rows: list[list[str]] = []
    now_iso = now.isoformat()
    for db, conn, _counts in discoveries:
        facets = _path_facets(db)
        rows = _select_rows(
            conn,
            statuses=statuses,
            session_id=session_id,
            only_overdue=only_overdue,
            only_waiting=only_waiting,
            now_iso=now_iso,
            limit=limit_per_db,
        )
        for r in rows:
            pred_id = r["id"] if verbose else r["id"].split("-")[-1]
            err = r["error_message"] or ""
            if not verbose:
                err = _truncate(err, 60)
            row_data = [
                _abbrev_task(facets["task"]),
                pred_id,
                _row_label(r, verbose=verbose),
                _format_number(r["predicted_price"]),
                _format_number(r["entry_price"]),
                _format_resolve_at(r["submitted_at"]),
                _format_resolve_at(r["resolve_at"]),
                r["status"] or "",
                err,
            ]
            if verbose:
                row_data.insert(2, facets["trial"])
            out_rows.append(row_data)
    _print_table(headers, out_rows)


# --resolve flow


def _prompt_yes_no(question: str, *, default_yes: bool = False) -> bool:
    """Single-shot Y/N prompt — no re-prompt loop.

    - empty input  → ``default_yes`` (just hit Enter to accept the default)
    - 'y' / 'yes'  → True
    - 'n' / 'no'   → False
    - anything else → False, with an explanatory message printed

    The loop is deliberately *not* re-entrant: a stray newline / paste /
    buffered keystroke on Windows can otherwise look like a "double
    prompt" and skip the user's actual answer. One read, one decision.
    """
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


def _run_resolve(
    discoveries: list[tuple[Path, sqlite3.Connection, dict[str, int]]],
    *,
    yes: bool,
    examples_root: Path,
) -> int:
    """For each ledger with ready > 0, locate its trial_dir and (with
    default-Y prompt or unconditionally if ``yes``) invoke
    resolve_deferred.main.

    Trial-dir lookup is two-tier: first read the ``trial_dir`` column
    on the ledger row (stamped at submit time post 2026-05-14); fall
    back to the ``examples/**/deferred_session.json`` reverse index for
    legacy DBs whose rows pre-date the column.

    Returns the count of trials whose resolver ran with exit code 0.
    """
    from openfinai_harbor.resolve_deferred import main as resolve_main

    index = _build_trial_dir_index(examples_root)
    ran_ok = 0
    for db, conn, counts in discoveries:
        if counts.get("ready", 0) <= 0:
            continue
        facets = _path_facets(db)
        # Primary: stamped column. Fallback: sidecar scan keyed by db path.
        trial_dir = _trial_dir_from_ledger(conn)
        if trial_dir is None:
            trial_dir = index.get(str(db.resolve()))
        if trial_dir is not None and not trial_dir.exists():
            # Pointer exists but the directory has been deleted —
            # genuine orphan. Distinguish from "never had a pointer."
            print(
                f"[resolve] {facets['task']}/{facets['trial']}: "
                f"trial_dir={trial_dir} no longer exists "
                f"(orphan ledger; consider scripts/maintenance/cleanup_stale_verifier_runs.py)."
            )
            continue
        if trial_dir is None:
            print(
                f"[resolve] {facets['task']}/{facets['trial']}: "
                f"ledger has {counts['ready']} ready row(s) but no matching "
                f"deferred_session.json under {examples_root}; resolve manually "
                f"by passing the trial_dir to resolve_deferred.py."
            )
            continue
        prompt = (
            f"Resolve {counts['ready']} row(s) for "
            f"{facets['task']}/{facets['trial']}?"
        )
        # Default Y: hitting Enter accepts. This is the integrated UX —
        # the inspector tells you exactly what's ready and offers to do
        # the work without a separate command.
        if not yes and not _prompt_yes_no(prompt, default_yes=True):
            print(f"[resolve] skipped {facets['task']}/{facets['trial']}")
            continue
        try:
            rc = resolve_main([str(trial_dir), "--verbose"])
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1
        except Exception as exc:  # pragma: no cover — defensive
            print(f"[resolve] {facets['task']}/{facets['trial']}: error {exc!r}")
            continue
        if rc == 0:
            ran_ok += 1
            print(f"[resolve] {facets['task']}/{facets['trial']}: OK")
        elif rc == 2:
            print(
                f"[resolve] {facets['task']}/{facets['trial']}: "
                f"nothing resolvable yet (provider returned no data)"
            )
        else:
            print(
                f"[resolve] {facets['task']}/{facets['trial']}: "
                f"resolver exited with code {rc}"
            )
    return ran_ok


# Main


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="inspect_ledger",
        description=(
            "Inspect deferred-prediction ledgers. "
            "Scans data/run_output/verifier/**/predictions.db. "
            "Interactive runs offer to resolve ready rows by default."
        ),
    )
    p.add_argument(
        "--root", type=Path, default=_DEFAULT_VERIFIER_ROOT,
        help="Root directory to scan for ledger DBs "
             f"(default: {_DEFAULT_VERIFIER_ROOT}).",
    )
    p.add_argument(
        "--examples-root", type=Path, default=_DEFAULT_EXAMPLES_ROOT,
        help="Root directory of example trials, for the reverse "
             f"ledger_path → trial_dir lookup used during resolution "
             f"(default: {_DEFAULT_EXAMPLES_ROOT}).",
    )
    p.add_argument(
        "--trial-dir", type=Path, default=None,
        help="Inspect just one trial — reads its deferred_session.json to "
             "locate the ledger DB and constrains to its session_id.",
    )
    p.add_argument(
        "--task", default=None,
        help="Filter to ledgers whose path contains this task_id.",
    )
    p.add_argument(
        "--session", dest="session_id", default=None,
        help="Filter rows to this session_id (works across all DBs).",
    )
    # Status / lifecycle filters (selecting any triggers row view).
    p.add_argument(
        "--waiting", action="store_true",
        help="Show pending rows whose resolve_at is still in the future.",
    )
    p.add_argument(
        "--ready", action="store_true",
        help="Show pending rows whose resolve_at <= now (actionable).",
    )
    p.add_argument(
        "--failed", action="store_true",
        help="Show status='failed' rows (terminal; needs intervention).",
    )
    p.add_argument(
        "--done", action="store_true",
        help="Show status='resolved' OR status='scored' rows.",
    )
    p.add_argument(
        "--limit", type=int, default=50,
        help="Max rows per ledger in row view (default 50).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Full pred_id + trial column + untruncated symbol/question + full error.",
    )
    # Interactive runs prompt before resolving ready rows; non-TTY runs do not.
    p.add_argument(
        "--no-resolve", action="store_true",
        help="Never prompt to resolve, even if ready>0 in an interactive TTY. "
             "Use for piped invocations where the prompt would block.",
    )
    p.add_argument(
        "--yes", action="store_true",
        help="Resolve all ready-row trials without prompting (CI / batch mode).",
    )
    return p.parse_args(argv)


def _statuses_from_args(args: argparse.Namespace) -> Optional[set[str]]:
    s: set[str] = set()
    if args.failed:
        s.add("failed")
    if args.done:
        s.add("resolved")
        s.add("scored")
    return s or None


def _open_discoveries(
    db_paths: Iterable[Path],
    *,
    now_iso: str,
) -> list[tuple[Path, sqlite3.Connection, dict[str, int]]]:
    out: list[tuple[Path, sqlite3.Connection, dict[str, int]]] = []
    for db in db_paths:
        try:
            conn = _open_ro(db)
        except sqlite3.Error as exc:
            print(f" warning: cannot open {db}: {exc}", file=sys.stderr)
            continue
        if not _is_ledger(conn):
            conn.close()
            continue
        try:
            counts = _summary_counts(conn, now_iso=now_iso)
        except sqlite3.Error as exc:
            print(f" warning: cannot read {db}: {exc}", file=sys.stderr)
            conn.close()
            continue
        out.append((db, conn, counts))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    session_id = args.session_id
    if args.trial_dir is not None:
        resolved = _resolve_trial_dir_to_ledger(args.trial_dir.resolve())
        if resolved is None:
            print(
                f"[inspect_ledger] no {_SIDECAR_NAME} under {args.trial_dir}; "
                f"trial may be offline (no deferred ledger).",
                file=sys.stderr,
            )
            return 1
        db_path, sid_from_sidecar = resolved
        db_paths = [db_path]
        session_id = session_id or sid_from_sidecar
    else:
        db_paths = _discover_ledgers(args.root.resolve())
        if args.task:
            db_paths = [p for p in db_paths if args.task in p.parts]

    if not db_paths:
        print(f"[inspect_ledger] no ledgers found under {args.root}", file=sys.stderr)
        return 1

    discoveries = _open_discoveries(db_paths, now_iso=now_iso)
    if not discoveries:
        print("[inspect_ledger] no ledger DBs were readable", file=sys.stderr)
        return 1

    try:
        statuses = _statuses_from_args(args)
        # Row view kicks in iff any filter narrows below the trial level.
        row_view = bool(
            statuses or args.ready or args.waiting
            or args.session_id or args.trial_dir
        )
        if row_view:
            _print_row_view(
                discoveries, now,
                statuses=statuses,
                session_id=session_id,
                only_overdue=args.ready,
                only_waiting=args.waiting,
                limit_per_db=args.limit,
                verbose=args.verbose,
            )
            total_ready = sum(d[2].get("ready", 0) for d in discoveries)
        else:
            total_ready = _print_summary_view(discoveries, now)

        # Resolve interactively by default; --no-resolve and --yes override.
        interactive = sys.stdin.isatty()
        should_resolve = (
            total_ready > 0
            and not args.no_resolve
            and (interactive or args.yes)
        )
        if should_resolve:
            print()
            _run_resolve(
                discoveries,
                yes=args.yes,
                examples_root=args.examples_root.resolve(),
            )
        elif total_ready > 0:
            n_trials = sum(
                1 for d in discoveries if d[2].get("ready", 0) > 0
            )
            hint_reason = (
                "non-interactive shell (stdin is not a TTY)"
                if not interactive else "—no-resolve"
            )
            print(
                f"\n{total_ready} row(s) ready to resolve across {n_trials} trial(s). "
                f"Skipped auto-resolve: {hint_reason}. Run with --yes or in a "
                f"TTY to resolve."
            )
    finally:
        for _, conn, _ in discoveries:
            conn.close()
    return 0


if __name__ == "__main__":
    # Support direct execution from a source checkout.
    _REPO = Path(__file__).resolve().parents[2]
    for _p in (_REPO / "src", _REPO):
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    raise SystemExit(main())
