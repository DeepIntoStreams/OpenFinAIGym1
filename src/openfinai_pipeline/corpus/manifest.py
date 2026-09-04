"""Dataset manifest schema for the corpus pipeline.

The manifest is the structured description of a downloaded dataset — it
assigns semantic roles to individual files so Phase 4 can locate ground
truth without filename-keyword heuristics.

The manifest lives on disk at ``<dataset_dir>/manifest.json`` and is the
authoritative source of truth for what a dataset contains. It is produced
post-download by the labeler LLM call (see ``codegen.label_dataset_artifacts``)
and consumed at Phase 4 evaluator assembly time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Semantic file-role vocabulary. ``features`` and ``ground_truth`` drive
# pairwise evaluation (forecasting); ``reference`` is for distributional
# evaluation (generative); ``metadata`` / ``auxiliary`` are informational.
ROLE_VOCAB: tuple[str, ...] = (
    "features",
    "ground_truth",
    "reference",
    "metadata",
    "auxiliary",
)


# Format-loader vocabulary. ``other`` is the escape hatch when the labeler
# cannot classify a file's format — downstream loaders treat it as
# unsupported.
FORMAT_VOCAB: tuple[str, ...] = (
    "npy",
    "npz",
    "csv",
    "parquet",
    "hdf5",
    "arrow",
    "feather",
    "json",
    "jsonl",
    "txt",
    "other",
)


# Manifest status values. ``labeled`` means the labeler produced confident
# role assignments; ``unresolved`` means the labeler couldn't decide and
# Phase 4 will hard-fail.
MANIFEST_STATUS_VALUES: tuple[str, ...] = (
    "labeled",
    "unresolved",
    "unlabeled_curated",
)


@dataclass
class ArtifactEntry:
    """One file within a dataset, with its semantic role and load info."""

    role: str
    path: str
    format: str
    shape: list[int] | None = None
    columns: list[str] | None = None
    dtype: str | None = None
    description: str = ""


@dataclass
class DatasetManifest:
    """Full on-disk description of a downloaded dataset.

    Fields fall into three groups:
      * Identity: ``name``, ``description``, ``dataset_kind``
      * Acquisition: ``downloaded_dataset_description``, ``used_by_papers``
      * Labeling: ``interaction_model``, ``manifest_status``,
        ``artifacts``, ``labeler_*``, plus ``has_ground_truth`` —
        derived from artifact roles (True iff any artifact has
        role=``ground_truth`` or role=``reference``), kept on the
        dataclass for back-compat with consumers that read the bool
        directly. The role-tagged ``artifacts`` list is the
        authoritative signal.

    ``extra`` preserves any unknown keys encountered at read time so that
    read/write round-trips are non-destructive.
    """

    name: str
    description: str = ""
    downloaded_dataset_description: str = ""
    dataset_kind: str = ""
    interaction_model: str = ""
    has_ground_truth: bool = False
    manifest_status: str = "unresolved"
    artifacts: list[ArtifactEntry] = field(default_factory=list)
    used_by_papers: list[dict[str, Any]] = field(default_factory=list)
    labeler_notes: str = ""
    labeler_provider: str = ""
    labeler_model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


_KNOWN_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "description",
        "downloaded_dataset_description",
        "dataset_kind",
        "interaction_model",
        "has_ground_truth",
        "manifest_status",
        "artifacts",
        "used_by_papers",
        "labeler_notes",
        "labeler_provider",
        "labeler_model",
    }
)


_ARTIFACT_KNOWN_KEYS: frozenset[str] = frozenset(
    {"role", "path", "format", "shape", "columns", "dtype", "description"}
)


def _artifact_from_dict(raw: Any) -> ArtifactEntry | None:
    if not isinstance(raw, dict):
        return None
    role = str(raw.get("role", "")).strip()
    path = str(raw.get("path", "")).strip()
    fmt = str(raw.get("format", "")).strip()
    if not role or not path or not fmt:
        return None
    shape_raw = raw.get("shape")
    shape: list[int] | None = None
    if isinstance(shape_raw, list):
        try:
            shape = [int(v) for v in shape_raw]
        except (TypeError, ValueError):
            shape = None
    columns_raw = raw.get("columns")
    columns: list[str] | None = None
    if isinstance(columns_raw, list):
        columns = [str(v) for v in columns_raw]
    dtype_raw = raw.get("dtype")
    dtype = str(dtype_raw) if dtype_raw is not None else None
    description = str(raw.get("description", "")).strip()
    return ArtifactEntry(
        role=role,
        path=path,
        format=fmt,
        shape=shape,
        columns=columns,
        dtype=dtype,
        description=description,
    )


def _artifact_to_dict(entry: ArtifactEntry) -> dict[str, Any]:
    out: dict[str, Any] = {
        "role": entry.role,
        "path": entry.path,
        "format": entry.format,
    }
    if entry.shape is not None:
        out["shape"] = list(entry.shape)
    if entry.columns is not None:
        out["columns"] = list(entry.columns)
    if entry.dtype is not None:
        out["dtype"] = entry.dtype
    if entry.description:
        out["description"] = entry.description
    return out


def manifest_from_dict(data: Any) -> DatasetManifest:
    """Tolerant reader: unknown keys are preserved in ``extra``.

    Missing or malformed fields fall back to defaults — this reader is
    intended to round-trip legacy manifests (``{name, used_by_papers}``
    shape) without loss.
    """
    if not isinstance(data, dict):
        return DatasetManifest(name="")
    artifacts_raw = data.get("artifacts", [])
    artifacts: list[ArtifactEntry] = []
    if isinstance(artifacts_raw, list):
        for item in artifacts_raw:
            entry = _artifact_from_dict(item)
            if entry is not None:
                artifacts.append(entry)
    used_by_raw = data.get("used_by_papers", [])
    used_by_papers: list[dict[str, Any]] = []
    if isinstance(used_by_raw, list):
        for item in used_by_raw:
            if isinstance(item, dict):
                used_by_papers.append(dict(item))
    extra: dict[str, Any] = {
        key: value for key, value in data.items() if key not in _KNOWN_TOP_LEVEL_KEYS
    }
    # ``has_ground_truth`` is derived from artifact roles. We honour an
    # explicit value in the input dict for backward compatibility with
    # older manifests written before this field became derived; if the
    # input is missing the field, we compute it from the role tags.
    if "has_ground_truth" in data:
        has_gt = bool(data.get("has_ground_truth"))
    else:
        has_gt = any(
            (a.role or "").strip() in ("ground_truth", "reference")
            for a in artifacts
        )
    return DatasetManifest(
        name=str(data.get("name", "")).strip(),
        description=str(data.get("description", "")).strip(),
        downloaded_dataset_description=str(
            data.get("downloaded_dataset_description", "")
        ).strip(),
        dataset_kind=str(data.get("dataset_kind", "")).strip(),
        interaction_model=str(data.get("interaction_model", "")).strip(),
        has_ground_truth=has_gt,
        manifest_status=(
            str(data.get("manifest_status", "")).strip() or "unresolved"
        ),
        artifacts=artifacts,
        used_by_papers=used_by_papers,
        labeler_notes=str(data.get("labeler_notes", "")),
        labeler_provider=str(data.get("labeler_provider", "")),
        labeler_model=str(data.get("labeler_model", "")),
        extra=extra,
    )


def manifest_to_dict(manifest: DatasetManifest) -> dict[str, Any]:
    """Serialize to a dict with stable top-level key order.

    ``extra`` keys are emitted after known keys so hand-edits survive
    round-trips.
    """
    out: dict[str, Any] = {
        "name": manifest.name,
        "description": manifest.description,
        "downloaded_dataset_description": manifest.downloaded_dataset_description,
        "dataset_kind": manifest.dataset_kind,
        "interaction_model": manifest.interaction_model,
        "has_ground_truth": manifest.has_ground_truth,
        "manifest_status": manifest.manifest_status,
        "artifacts": [_artifact_to_dict(a) for a in manifest.artifacts],
        "used_by_papers": [dict(item) for item in manifest.used_by_papers],
        "labeler_notes": manifest.labeler_notes,
        "labeler_provider": manifest.labeler_provider,
        "labeler_model": manifest.labeler_model,
    }
    for key, value in manifest.extra.items():
        if key not in out:
            out[key] = value
    return out


def dataset_label_schema() -> dict[str, Any]:
    """JSON Schema for the labeler LLM's structured output.

    Covers labeler-owned fields: ``manifest_status``, ``artifacts``,
    ``labeler_notes``. The remaining manifest fields (including
    ``has_ground_truth``) are filled server-side: ``has_ground_truth``
    is derived from the artifact role tags (True iff at least one
    artifact has role=``ground_truth`` or role=``reference``) and the
    rest come from known dataset metadata so the LLM can't overwrite
    them with hallucinated values.
    """
    return {
        "title": "DatasetLabelerOutput",
        "description": (
            "Post-download labeler output assigning semantic roles to each "
            "file in a freshly materialized dataset."
        ),
        "type": "object",
        "properties": {
            "labeler_notes": {"type": "string"},
            "artifacts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": list(ROLE_VOCAB)},
                        "path": {"type": "string"},
                        "format": {"type": "string", "enum": list(FORMAT_VOCAB)},
                        "shape": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "dtype": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["role", "path", "format"],
                    "additionalProperties": False,
                },
            },
            "manifest_status": {
                "type": "string",
                "enum": ["labeled", "unresolved"],
            },
        },
        "required": [
            "labeler_notes",
            "artifacts",
            "manifest_status",
        ],
        "additionalProperties": False,
    }


def infer_format_from_suffix(path: str) -> str:
    """Map a file path's extension to a value in ``FORMAT_VOCAB``.

    Used as a fallback when the labeler doesn't classify a real file.
    Unknown suffixes map to ``"other"``.
    """
    lowered = path.lower()
    # Order matters: check compound suffixes first.
    if lowered.endswith(".jsonl"):
        return "jsonl"
    if lowered.endswith(".json"):
        return "json"
    if lowered.endswith(".parquet") or lowered.endswith(".pq"):
        return "parquet"
    if lowered.endswith(".npz"):
        return "npz"
    if lowered.endswith(".npy"):
        return "npy"
    if lowered.endswith(".csv") or lowered.endswith(".tsv"):
        return "csv"
    if lowered.endswith(".h5") or lowered.endswith(".hdf5"):
        return "hdf5"
    if lowered.endswith(".feather"):
        return "feather"
    if lowered.endswith(".arrow") or lowered.endswith(".ipc"):
        return "arrow"
    if lowered.endswith(".txt") or lowered.endswith(".md"):
        return "txt"
    return "other"


_FORMAT_ALIASES: dict[str, str] = {
    "pq": "parquet",
    "h5": "hdf5",
    "tsv": "csv",
    "text": "txt",
    "plain": "txt",
    "ipc": "arrow",
    "arrow_ipc": "arrow",
    "feather_v2": "feather",
}


def canonicalize_format(fmt: str) -> str:
    """Normalize labeler-provided format values against ``FORMAT_VOCAB``.

    Handles common aliases (``pq`` → ``parquet``, ``h5`` → ``hdf5``,
    ``tsv`` → ``csv``). Unrecognized values collapse to ``"other"``.
    """
    lowered = str(fmt or "").strip().lower()
    if not lowered:
        return "other"
    if lowered in FORMAT_VOCAB:
        return lowered
    if lowered in _FORMAT_ALIASES:
        return _FORMAT_ALIASES[lowered]
    return "other"
