"""Build a deterministic, train-only SFT corpus.

Protected benchmark tasks never enter training. Trials remain grouped by task,
and generalization is measured externally through Harbor evaluation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from datasets import Dataset, DatasetDict


DEFAULT_TEST_TASKS: tuple[str, ...] = (
    "1_commodity_logreturn_forecasting",
    "1_crypto_variant3_logreturn_forecasting",
    "1_equity_variant2_logreturn_forecasting",
    "1_treasury_variant2_forecasting",
    "1_fx_variant4_logreturn_forecasting",
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def available_task_names(tasks_root: str | Path) -> list[str]:
    root = Path(tasks_root)
    return sorted(path.name for path in root.iterdir() if path.is_dir() and path.name != "_template")


def normalize_exported_task_name(raw_task_name: str, known_tasks: Iterable[str]) -> str:
    """Resolve a possibly-truncated task name to its canonical form.

    Harbor sometimes truncates very long task names; this helper finds
    the unique installed task that the truncated form matches. Raises
    ``ValueError`` if zero or multiple tasks match.
    """
    if raw_task_name in known_tasks:
        return raw_task_name

    prefix_matches = [task for task in known_tasks if task.startswith(raw_task_name)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    suffix_matches = [task for task in known_tasks if raw_task_name.startswith(task)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    raise ValueError(
        f"Could not normalize task name {raw_task_name!r}; matches={prefix_matches or suffix_matches or []}"
    )


@dataclass
class CorpusSplitManifest:
    """Manifest serialised next to the train-only SFT corpus."""

    created_at_utc: str
    tasks_root: str
    protected_test_tasks: list[str]
    train_tasks: list[str]
    rows_by_task: dict[str, int]
    row_counts: dict[str, int]
    n_dropped_test_task_rows: int = 0
    dropped_test_task_breakdown: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_train_only_corpus(
    *,
    rows: Sequence[dict[str, Any]],
    tasks_root: str | Path,
    test_tasks: Iterable[str] = DEFAULT_TEST_TASKS,
) -> tuple[DatasetDict, CorpusSplitManifest]:
    """Drop test-task rows and assemble a train-only ``DatasetDict``.

    Steps:

    1. Canonicalise task names via :func:`normalize_exported_task_name`.
    2. Drop any rows whose task is in ``test_tasks`` (the predefined
       held-out benchmark). The count is reported in the manifest so
       accidental collection on test tasks is visible.
    3. Sort within-task rows by ``trial_name`` so re-runs emit identical
       corpora; concatenate by sorted task name.

    Args:
        rows: Raw row dicts. Each row must have a ``"task"`` key.
        tasks_root: Path to the ``tasks/`` directory; used to canonicalise
            task names (Harbor sometimes truncates them).
        test_tasks: Tasks that MUST be excluded from the SFT corpus.
            Defaults to :data:`DEFAULT_TEST_TASKS`.

    Returns:
        ``(DatasetDict({"train": ...}), manifest)``. The returned
        ``DatasetDict`` contains only a ``train`` split; there is no
        in-corpus held-out split.
    """
    if not rows:
        raise ValueError("build_train_only_corpus: rows is empty")

    known_tasks = available_task_names(tasks_root)
    protected = {name for name in dict.fromkeys(test_tasks)}

    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dropped_breakdown: dict[str, int] = defaultdict(int)

    for row in rows:
        task_name = str(row.get("task", "")).strip()
        if not task_name:
            continue
        normalized = normalize_exported_task_name(task_name, known_tasks)
        if normalized in protected:
            dropped_breakdown[normalized] += 1
            continue
        normalized_row = dict(row)
        normalized_row["task"] = normalized
        normalized_row["task_dir"] = str(Path(tasks_root) / normalized)
        rows_by_task[normalized].append(normalized_row)

    if not rows_by_task:
        raise ValueError(
            "build_train_only_corpus: every input row belonged to a protected "
            f"test task ({sorted(dropped_breakdown)!r}); nothing left to train on."
        )

    # Stable within-task ordering so reruns produce byte-identical corpora.
    for task in rows_by_task:
        rows_by_task[task].sort(key=lambda r: str(r.get("trial_name", "")))

    train_tasks = sorted(rows_by_task)
    train_rows: list[dict[str, Any]] = []
    for task in train_tasks:
        train_rows.extend(rows_by_task[task])

    dataset = DatasetDict({"train": Dataset.from_list(train_rows)})

    manifest = CorpusSplitManifest(
        created_at_utc=_utc_timestamp(),
        tasks_root=str(Path(tasks_root).resolve()),
        protected_test_tasks=sorted(protected),
        train_tasks=train_tasks,
        rows_by_task={task: len(rs) for task, rs in rows_by_task.items()},
        row_counts={split: len(ds) for split, ds in dataset.items()},
        n_dropped_test_task_rows=sum(dropped_breakdown.values()),
        dropped_test_task_breakdown=dict(dropped_breakdown),
    )
    return dataset, manifest


__all__ = [
    "DEFAULT_TEST_TASKS",
    "CorpusSplitManifest",
    "available_task_names",
    "build_train_only_corpus",
    "normalize_exported_task_name",
]
