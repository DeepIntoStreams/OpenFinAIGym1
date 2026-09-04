"""
Filter papers/tasks by dataset download availability.

Provides reusable utilities for both Phase 3 (reward construction) and
Phase 4 (benchmark task construction) to skip papers whose datasets
were not successfully downloaded.
"""

import logging
from pathlib import Path
from typing import Any

from openfinai_pipeline.utils.logging import log_detail

logger = logging.getLogger(__name__)
_TAG = "[dataset_filter]"


def _dataset_entry_paths(dataset_entry: dict[str, Any]) -> list[str]:
    return [
        str(path).strip()
        for path in dataset_entry.get("dataset_paths", [])
        if str(path).strip()
    ]


def _dataset_entry_unready_reason(dataset_entry: dict[str, Any]) -> str:
    """Return ``""`` when the entry is ready, or a short tag explaining why not.

    Single gate: download success — ``download_successful=True`` and every
    relative path in ``dataset_paths`` exists on disk. Per-task ``load.py``
    generation is a Phase 4 concern (see
    :mod:`openfinai_pipeline.benchmark.loader`), so a missing loader does
    not block dataset readiness here.
    """
    if not bool(dataset_entry.get("download_successful", False)):
        return "download_failed"
    dataset_paths = _dataset_entry_paths(dataset_entry)
    if not dataset_paths:
        return "no_dataset_paths"
    if not all(Path(path).exists() for path in dataset_paths):
        return "missing_files"
    return ""


def _dataset_entry_is_ready(dataset_entry: dict[str, Any]) -> bool:
    return _dataset_entry_unready_reason(dataset_entry) == ""


def check_paper_datasets_ready(
    scope_id: str,
    datasets: list[dict[str, str]],
    datasets_root: Path,
    curated_datasets_root: Path | None = None,
) -> tuple[bool, list[str]]:
    """Check whether at least one listed dataset for a paper is ready.

    Returns ``(has_at_least_one_ready, missing)`` where each missing entry
    is annotated with the failure reason (e.g.
    ``"DatasetA(missing_files)"``) so phase-skip logs surface
    actionable diagnostics.
    """
    del scope_id, datasets_root, curated_datasets_root
    if not datasets:
        return True, []

    missing: list[str] = []
    ready_count = 0
    for ds in datasets:
        name = str(ds.get("name", "")).strip()
        if not name:
            continue
        reason = _dataset_entry_unready_reason(ds)
        if not reason:
            ready_count += 1
            continue
        missing.append(f"{name}({reason})")

    return ready_count > 0, missing


def filter_task_summaries_by_dataset_availability(
    task_summaries: list[tuple[str, dict[str, Any]]],
    datasets_root: Path,
    curated_datasets_root: Path | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Filter task_summaries to only papers with at least one ready dataset."""
    filtered: list[tuple[str, dict[str, Any]]] = []
    skipped: list[tuple[str, list[str]]] = []
    for task_dir, summary in task_summaries:
        scope_id = str(summary.get("_scope_id", "default")).strip() or "default"
        datasets = summary.get("datasets", []) or []
        has_ready_dataset, missing = check_paper_datasets_ready(
            scope_id,
            datasets,
            datasets_root,
            curated_datasets_root=curated_datasets_root,
        )
        if has_ready_dataset:
            filtered.append((task_dir, summary))
        else:
            skipped.append((task_dir, missing))

    logger.info(
        "%s filtered task summaries by dataset availability: %d/%d passed with at least one ready dataset, %d skipped",
        _TAG,
        len(filtered),
        len(task_summaries),
        len(skipped),
    )

    if skipped:
        detail_lines = [f"  {td}: {','.join(ms)}" for td, ms in skipped]
        detail_msg = f"{_TAG} skipped papers (no listed datasets ready):\n" + "\n".join(
            detail_lines
        )
        log_detail(logger, detail_msg)

    return filtered
