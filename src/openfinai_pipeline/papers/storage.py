import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openfinai_pipeline.papers.schemas import (
    JudgeDecision,
    PaperRecord,
    PaperTaskBinding,
    SeenRejectedEntry,
    TaskIndex,
    TaskIndexEntry,
    TradingTriageRecord,
)


class ResearchStore:
    """Persist accepted-paper artifacts in a scope-aware paperN layout."""

    def __init__(self, root_dir: str) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def persist_accepted_paper(
        self,
        *,
        run_id: str,
        scope_id: str,
        paper: PaperRecord,
        decision: JudgeDecision,
        source_pdf_path: str | None = None,
        preserve_source_pdf: bool = False,
    ) -> str:
        target, entry, index_path, index = self._prepare_target(
            run_id=run_id, scope_id=scope_id, paper=paper
        )
        (target / "paper.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "scope_id": scope_id,
                    "paper": paper.model_dump(),
                    "decision": decision.model_dump(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        synced_pdf = self._sync_source_pdf(target, source_pdf_path, preserve_source=preserve_source_pdf)
        self._save_index(index_path, index)
        return f"{scope_id}/{entry.paper_dir}"

    def find_task_dir(self, *, scope_id: str, paper_id: str) -> Path | None:
        """Return the on-disk paperN/ for an *accepted* paper, else None.

        Filters by `entry.label == "accepted"`: rejected entries (if any
        ever land in `entries`, e.g. from external tooling) cannot
        accidentally surface a phantom directory path.
        """
        scope_root = self._root / scope_id
        index_path = scope_root / "index.json"
        index = self._load_index(index_path)
        entry = self._find_entry(index, paper_id)
        if entry is None:
            return None
        return scope_root / entry.paper_dir

    def find_task_dir_by_raw_payload_hash(
        self,
        *,
        scope_id: str,
        raw_payload_hash: str,
    ) -> Path | None:
        scope_root = self._root / scope_id
        if not scope_root.exists():
            return None
        for paper_dir in sorted(
            [p for p in scope_root.iterdir() if p.is_dir() and p.name.startswith("paper")]
        ):
            paper_json_path = paper_dir / "paper.json"
            if not paper_json_path.exists():
                continue
            try:
                payload = json.loads(paper_json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            paper_doc = payload.get("paper", {})
            if not isinstance(paper_doc, dict):
                continue
            if str(paper_doc.get("raw_payload_hash", "")).strip() == raw_payload_hash:
                return paper_dir
        return None

    def list_paper_ids(self, *, scope_id: str) -> list[str]:
        """Paper IDs for *accepted* papers in this scope.

        Filters by `entry.label == "accepted"` so the manual-import dedupe
        path (which uses this) cannot collide with a rejected entry that
        somehow shares an ID.
        """
        scope_root = self._root / scope_id
        index_path = scope_root / "index.json"
        index = self._load_index(index_path)
        return [
            entry.paper_id
            for entry in index.entries
            if entry.label == "accepted"
        ]

    def list_seen_paper_ids(
        self, *, scope_id: str
    ) -> tuple[set[str], set[str]]:
        """Return (accepted_ids, rejected_ids) for this scope's prior runs.

        Phase 1 scrape uses this post-collection to skip work on papers
        already judged. Per-scope partitioning is intentional: a paper
        rejected under scope A may legitimately be accepted under scope B
        (different queries/categories/threshold).
        """
        scope_root = self._root / scope_id
        index_path = scope_root / "index.json"
        index = self._load_index(index_path)
        accepted = {
            entry.paper_id
            for entry in index.entries
            if entry.label == "accepted"
        }
        rejected = {entry.paper_id for entry in index.seen_rejected}
        return accepted, rejected

    def record_rejected(
        self,
        *,
        scope_id: str,
        run_id: str,
        items: list[tuple[str, str, str]],
    ) -> None:
        """Append/update SeenRejectedEntry rows for one judging pass.

        ``items`` is a list of ``(paper_id, model, reason)`` tuples — the
        fields needed to debug "why was this skipped on rerun?". Updates
        existing entries in place rather than appending duplicates.

        Skips any paper_id already present in ``index.entries`` (accepted):
        the accepted record wins and the two lists stay disjoint. This
        protects against a rare race where the same paper is judged
        differently under ``--overwrite`` than its prior persisted state.
        """
        if not items:
            return
        scope_root = self._root / scope_id
        scope_root.mkdir(parents=True, exist_ok=True)
        index_path = scope_root / "index.json"
        index = self._load_index(index_path)

        accepted_ids = {
            entry.paper_id
            for entry in index.entries
            if entry.label == "accepted"
        }
        existing_rejected: dict[str, SeenRejectedEntry] = {
            entry.paper_id: entry for entry in index.seen_rejected
        }
        now = datetime.now(tz=timezone.utc).isoformat()
        changed = False
        for paper_id, model, reason in items:
            if paper_id in accepted_ids:
                continue
            trimmed_reason = (reason or "")[:500]
            existing = existing_rejected.get(paper_id)
            if existing is None:
                entry = SeenRejectedEntry(
                    paper_id=paper_id,
                    last_run_id=run_id,
                    last_judged_at=now,
                    last_reason=trimmed_reason,
                    last_model=model or "",
                )
                index.seen_rejected.append(entry)
                existing_rejected[paper_id] = entry
                changed = True
            else:
                existing.last_run_id = run_id
                existing.last_judged_at = now
                existing.last_reason = trimmed_reason
                existing.last_model = model or ""
                changed = True
        if changed:
            self._save_index(index_path, index)

    def persist_paper_summary(
        self,
        *,
        paper_dir: Path,
        summary_payload: dict[str, Any],
    ) -> None:
        paper_json_path = paper_dir / "paper.json"
        paper_doc: dict[str, Any] = {}
        if paper_json_path.exists():
            paper_doc = json.loads(paper_json_path.read_text(encoding="utf-8"))
        paper_doc["summary"] = summary_payload
        paper_json_path.write_text(
            json.dumps(paper_doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def persist_trading_triage(
        self,
        *,
        paper_dir: Path,
        record: TradingTriageRecord,
    ) -> Path:
        """Write the triage sidecar as ``trading_triage.json`` next to paper.json.

        Atomic via tmp + rename so a partial write never leaves a corrupt sidecar.
        Returns the written path.
        """
        target = paper_dir / "trading_triage.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(record.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(target)
        return target

    def persist_paper_task(
        self,
        *,
        paper_dir: Path,
        binding: PaperTaskBinding,
    ) -> None:
        """Read-modify-write ``paper.json["task"]`` with the unified binding shape.

        Both Phase 4 (auto_generated) and Phase 1c (curated_routed) call this to
        record the paper -> task pointer; consumers dispatch on ``binding.kind``.
        """
        paper_json_path = paper_dir / "paper.json"
        paper_doc: dict[str, Any] = {}
        if paper_json_path.exists():
            paper_doc = json.loads(paper_json_path.read_text(encoding="utf-8"))
        paper_doc["task"] = binding.model_dump(mode="json", exclude_none=True)
        paper_json_path.write_text(
            json.dumps(paper_doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def iter_task_dirs(self) -> list[Path]:
        task_dirs: list[Path] = []
        for scope_dir in sorted([p for p in self._root.iterdir() if p.is_dir()]):
            for paper_dir in sorted(
                [
                    p
                    for p in scope_dir.iterdir()
                    if p.is_dir() and p.name.startswith("paper")
                ]
            ):
                task_dirs.append(paper_dir)
        return task_dirs

    def count_accepted_for_scope(self, scope_id: str) -> int:
        """Return number of previously accepted papers for the given scope.

        Filters by `entry.label == "accepted"` defensively — if any
        non-accepted entry ever lands in `entries`, it must not inflate
        the accept-cap math at `_download_accepted_pdfs`.
        """
        index_path = self._root / scope_id / "index.json"
        index = self._load_index(index_path)
        return sum(
            1 for entry in index.entries if entry.label == "accepted"
        )

    def count_routes_for_scope(self, scope_id: str) -> int:
        """Return number of papers in this scope already routed by Phase 1c.

        A "route" is a `trading_triage.json` sidecar whose `status` is
        `routed` or `partial_match` — both produce an overlay under
        `tasks/routed/<id>/` and a `paper.json["task"]` binding. Sidecars
        with `no_match` or `novel_task_required` write no artifact and
        do not count. Drives Phase 1c's scope-aware `routes_written_cap`.
        """
        scope_root = self._root / scope_id
        if not scope_root.exists():
            return 0
        routed_statuses = {"routed", "partial_match"}
        count = 0
        for paper_dir in scope_root.iterdir():
            if not (paper_dir.is_dir() and paper_dir.name.startswith("paper")):
                continue
            triage_path = paper_dir / "trading_triage.json"
            if not triage_path.exists():
                continue
            try:
                record = json.loads(triage_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if str(record.get("status", "")).strip().lower() in routed_statuses:
                count += 1
        return count

    def _prepare_target(
        self, *, run_id: str, scope_id: str, paper: PaperRecord
    ) -> tuple[Path, TaskIndexEntry, Path, TaskIndex]:
        scope_root = self._root / scope_id
        scope_root.mkdir(parents=True, exist_ok=True)
        index_path = scope_root / "index.json"
        index = self._load_index(index_path)
        existing = self._find_entry(index, paper.paper_id)
        if existing is None:
            paper_id_num = index.next_paper_id
            paper_dir = f"paper{paper_id_num}"
            index.next_paper_id += 1
            entry = TaskIndexEntry(
                paper_id_num=paper_id_num,
                paper_dir=paper_dir,
                run_id=run_id,
                scope_id=scope_id,
                paper_id=paper.paper_id,
                title=paper.title,
                label="accepted",
            )
            index.entries.append(entry)
        else:
            entry = existing
            entry.run_id = run_id
            entry.scope_id = scope_id
            entry.title = paper.title
            entry.label = "accepted"
            entry.updated_at = datetime.now(tz=timezone.utc).isoformat()

        # Promotion: if this paper_id was previously logged as rejected
        # (different run, or this run under --overwrite), drop it from
        # seen_rejected so the two lists stay disjoint.
        index.seen_rejected = [
            sr for sr in index.seen_rejected if sr.paper_id != paper.paper_id
        ]

        target = scope_root / entry.paper_dir
        target.mkdir(parents=True, exist_ok=True)
        return target, entry, index_path, index

    def _load_index(self, index_path: Path) -> TaskIndex:
        if not index_path.exists():
            return TaskIndex()
        return TaskIndex.model_validate_json(index_path.read_text(encoding="utf-8"))

    def _save_index(self, index_path: Path, index: TaskIndex) -> None:
        index_path.write_text(index.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _sync_source_pdf(task_dir: Path, source_pdf_path: str | None, *, preserve_source: bool = False) -> bool:
        if not source_pdf_path:
            return False
        source = Path(source_pdf_path)
        if not source.exists() or not source.is_file():
            return False
        target = task_dir / "source.pdf"
        try:
            if target.exists():
                target.unlink(missing_ok=True)
            if preserve_source:
                shutil.copy2(str(source), str(target))
            else:
                shutil.move(str(source), str(target))
            return True
        except OSError:
            return False

    @staticmethod
    def _find_entry(index: TaskIndex, paper_id: str) -> TaskIndexEntry | None:
        """Find an *accepted* entry by paper_id.

        Filters by `entry.label == "accepted"` so `_prepare_target`'s slot
        reuse cannot mistakenly return a non-accepted entry. (Today no code
        path writes non-accepted entries to `entries`, but the defensive
        filter prevents a future regression.)
        """
        for entry in index.entries:
            if entry.paper_id == paper_id and entry.label == "accepted":
                return entry
        return None
