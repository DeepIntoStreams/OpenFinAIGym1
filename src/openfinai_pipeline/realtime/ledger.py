"""SQLite-backed prediction ledger for deferred evaluation.

Stores predictions submitted by agents during realtime trading sessions.
A separate resolver later fills in ground truth and rewards.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS predictions (
    id               TEXT    PRIMARY KEY,
    session_id       TEXT    NOT NULL,
    provider         TEXT    NOT NULL,
    symbol           TEXT    NOT NULL,
    horizon_minutes  INTEGER NOT NULL,
    resolution_interval TEXT,
    submitted_at     TEXT    NOT NULL,
    resolve_at       TEXT    NOT NULL,
    direction        TEXT    NOT NULL,
    predicted_price  REAL,
    entry_price      REAL    NOT NULL,
    snapshot         TEXT,
    exit_price       REAL,
    actual_return    REAL,
    ground_truth     TEXT,
    rewards          TEXT,
    status           TEXT    NOT NULL DEFAULT 'pending',
    resolved_at      TEXT,
    scored_at        TEXT,
    error_message    TEXT,
    trial_dir        TEXT
);
CREATE INDEX IF NOT EXISTS idx_status_resolve
    ON predictions(status, resolve_at);
CREATE INDEX IF NOT EXISTS idx_session
    ON predictions(session_id);
CREATE INDEX IF NOT EXISTS idx_symbol_time
    ON predictions(symbol, submitted_at);
"""

# Columns ADDed in-place if missing from an existing DB; existing rows
# get NULL. SQLite ``ADD COLUMN`` is O(1) on metadata, so cheaper than
# rebuilding the table.
_REQUIRED_COLUMNS: dict[str, str] = {
    "predicted_price": "REAL",
    "trial_dir": "TEXT",
    # Bar interval the resolver fetches the exit price at (= the task's
    # data_resolution). NULL on legacy rows / event-shape predictions →
    # resolver falls back to "1m". Nullable so ADD COLUMN works in-place.
    "resolution_interval": "TEXT",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PredictionLedger:
    """CRUD interface over the predictions SQLite database."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False is required because the curated
        # verifier (FastAPI) creates the ledger in on_startup but
        # services requests on a different thread. Concurrent access
        # is still safe: the verifier serialises writes under
        # ``state.score_lock``, and the synchronous CLI uses a single
        # thread anyway.
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate_columns()

    def _migrate_columns(self) -> None:
        """Idempotent column-add migration for older ledger databases.

        ``CREATE TABLE IF NOT EXISTS`` is a no-op when the table already
        exists, so a column added to ``_SCHEMA`` after the DB was first
        created would never be applied. We list the post-creation columns
        in ``_REQUIRED_COLUMNS`` and ``ALTER TABLE ... ADD COLUMN`` any
        that are missing. New columns must be nullable (no NOT NULL,
        no non-constant default) — SQLite's ADD COLUMN forbids those.
        """
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        for col, sql_type in _REQUIRED_COLUMNS.items():
            if col not in existing:
                self._conn.execute(
                    f"ALTER TABLE predictions ADD COLUMN {col} {sql_type}"
                )
        self._conn.commit()

    # Write

    def submit(self, record: dict[str, Any]) -> str:
        """Store a prediction and return its id.

        ``predicted_price`` is optional — direction-only legacy
        submissions store NULL there and the resolver/scorer NaN-skips
        price metrics for those rows.
        """
        pred_id = str(uuid4())
        self._conn.execute(
            """INSERT INTO predictions
               (id, session_id, provider, symbol, horizon_minutes,
                resolution_interval, submitted_at, resolve_at, direction,
                predicted_price, entry_price, snapshot, status,
                trial_dir)
               VALUES (?,?,?,?,?, ?,?,?,?, ?,?,?,?, ?)""",
            (
                pred_id,
                record["session_id"],
                record.get("provider", ""),
                record["symbol"],
                record["horizon_minutes"],
                record.get("resolution_interval"),
                record["submitted_at"].isoformat()
                if isinstance(record["submitted_at"], datetime)
                else str(record["submitted_at"]),
                record["resolve_at"].isoformat()
                if isinstance(record["resolve_at"], datetime)
                else str(record["resolve_at"]),
                record["direction"],
                record.get("predicted_price"),
                record["entry_price"],
                json.dumps(record.get("snapshot", {})),
                "pending",
                record.get("trial_dir"),
            ),
        )
        self._conn.commit()
        return pred_id

    def mark_resolved(
        self,
        pred_id: str,
        exit_price: float | None,
        actual_return: float | None,
        ground_truth: dict[str, Any] | None = None,
    ) -> None:
        """Fill in ground truth for a pending prediction.

        ``exit_price`` and ``actual_return`` are allowed to be ``None``
        for event-resolution rows (Polymarket etc.) where the concept
        of a continuous price doesn't apply — the binary outcome lives
        in the ``ground_truth`` JSON instead.
        """
        self._conn.execute(
            """UPDATE predictions
               SET exit_price=?, actual_return=?, ground_truth=?,
                   status='resolved', resolved_at=?
               WHERE id=? AND status='pending'""",
            (
                exit_price,
                actual_return,
                json.dumps(ground_truth or {}),
                _utcnow_iso(),
                pred_id,
            ),
        )
        self._conn.commit()

    def mark_scored(self, pred_id: str, rewards: dict[str, float]) -> None:
        """Attach computed rewards to a resolved prediction."""
        self._conn.execute(
            """UPDATE predictions
               SET rewards=?, status='scored', scored_at=?
               WHERE id=? AND status='resolved'""",
            (json.dumps(rewards), _utcnow_iso(), pred_id),
        )
        self._conn.commit()

    def mark_failed(self, pred_id: str, error: str) -> None:
        self._conn.execute(
            """UPDATE predictions SET status='failed', error_message=?
               WHERE id=?""",
            (error, pred_id),
        )
        self._conn.commit()

    # Read

    def get_pending(self, before: datetime, limit: int = 100) -> list[dict[str, Any]]:
        """Return pending predictions whose resolve_at is before *before*."""
        rows = self._conn.execute(
            """SELECT * FROM predictions
               WHERE status='pending' AND resolve_at <= ?
               ORDER BY resolve_at
               LIMIT ?""",
            (before.isoformat(), limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_resolved_unscored(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return resolved predictions that have not been scored yet."""
        rows = self._conn.execute(
            """SELECT * FROM predictions
               WHERE status='resolved'
               ORDER BY resolved_at
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM predictions WHERE session_id=? ORDER BY submitted_at",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_summary(
        self,
        session_id: str | None = None,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate counts by status, optionally filtered."""
        where_parts: list[str] = []
        params: list[Any] = []
        if session_id:
            where_parts.append("session_id=?")
            params.append(session_id)
        if symbol:
            where_parts.append("symbol=?")
            params.append(symbol)
        where = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        rows = self._conn.execute(
            f"SELECT status, COUNT(*) as cnt FROM predictions{where} GROUP BY status",
            params,
        ).fetchall()
        counts = {r["status"]: r["cnt"] for r in rows}
        counts["total"] = sum(counts.values())
        return counts

    # Lifecycle

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PredictionLedger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
