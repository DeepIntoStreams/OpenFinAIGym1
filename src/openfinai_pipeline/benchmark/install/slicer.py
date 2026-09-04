"""Demultiplex validated loader output into agent and verifier HDF5 files.

``dataset.h5`` contains training data and test features; verifier-only
``test_ground_truth.h5`` contains held-out targets. Unconditional generative
references may appear in both because they have no held-out target. The loader
owns the split policy; this module validates and serializes it without
re-splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

__all__ = [
    "SliceResult",
    "SliceShapeError",
    "slice_b_shape",
    "write_dataset_h5",
    "write_test_gt_h5",
]


# Errors


class SliceShapeError(ValueError):
    """Raised when the loader output does not satisfy the B-shape contract.

    Distinguished from generic ``ValueError`` so callers (``install_task``
    and dry-fire) can surface a specific failure to ops without
    swallowing legitimate validation issues.
    """


# Public types


@dataclass
class SliceResult:
    """Outcome of :func:`slice_b_shape`.

    Attributes:
        shape: ``"b_shape"`` (with train + test bundles) or ``"reference"``
            (unconditional generative).
        agent_payload: The dict that should be written to ``dataset.h5``. For
            B-shape this is the input minus the ``test/ground_truth`` leaf.
            For reference-shape it is the input unchanged.
        held_out: The ndarray (or ``{sym: ndarray}`` dict) that should be
            written to ``test_ground_truth.h5``. For unconditional generative
            this is the full reference (agent and verifier see the same data
            because there is no held-out distinction).
        split_policy: The policy string the LLM declared
            (``"chronological_80_20"``, ``"paper_dates"``, ``"random_80_20"``,
            ``"no_split"``, ...).
    split_metadata:
        The optional metadata dict (e.g. paper-stated split boundaries).
    """

    shape: str
    agent_payload: Mapping[str, Any]
    held_out: Any
    split_policy: str
    split_metadata: dict[str, Any] | None


# Slicer


_VALID_SHAPES: tuple[str, ...] = ("b_shape", "reference")
_VALID_LEAF_KEYS: frozenset[str] = frozenset({"features", "ground_truth"})


def slice_b_shape(loader_output: Mapping[str, Any]) -> SliceResult:
    """Demux the LLM loader's output dict into agent + verifier halves.

    Validates the input against the B-shape (or reference-shape) contract
    and returns a :class:`SliceResult` whose ``agent_payload`` and
    ``held_out`` fields are the inputs to :func:`write_dataset_h5` and
    :func:`write_test_gt_h5` respectively.

    Raises:
        SliceShapeError: If the dict is missing required keys, has unexpected
            leaf types, or mixes B-shape with reference-shape sentinels.
    """
    if not isinstance(loader_output, Mapping):
        raise SliceShapeError(
            f"loader output must be a mapping; got {type(loader_output).__name__}"
        )

    split_policy = str(loader_output.get("split_policy", "")).strip()
    if not split_policy:
        raise SliceShapeError(
            "loader output is missing the required 'split_policy' field"
        )
    split_metadata = loader_output.get("split_metadata")
    if split_metadata is not None and not isinstance(split_metadata, Mapping):
        raise SliceShapeError(
            "'split_metadata' must be a mapping or None; got "
            f"{type(split_metadata).__name__}"
        )

    # Reference-shape (unconditional generative) ------------------------
    if "reference" in loader_output:
        if "train" in loader_output or "test" in loader_output:
            raise SliceShapeError(
                "loader output mixes 'reference' with 'train'/'test' keys; "
                "use one shape or the other"
            )
        if split_policy != "no_split":
            raise SliceShapeError(
                f"reference-shape requires split_policy='no_split'; got {split_policy!r}"
            )
        reference = loader_output["reference"]
        _validate_leaf("reference", reference)
        agent_payload = {"reference": reference}
        return SliceResult(
            shape="reference",
            agent_payload=agent_payload,
            held_out=reference,  # agent and verifier see the same data
            split_policy=split_policy,
            split_metadata=dict(split_metadata) if split_metadata else None,
        )

    # B-shape (forecasting / conditional generative) --------------------
    for required in ("train", "test"):
        if required not in loader_output:
            raise SliceShapeError(
                f"B-shape loader output is missing the required {required!r} bundle"
            )
        bundle = loader_output[required]
        if not isinstance(bundle, Mapping):
            raise SliceShapeError(
                f"{required!r} bundle must be a mapping; got {type(bundle).__name__}"
            )
        for leaf_key in ("features", "ground_truth"):
            if leaf_key not in bundle:
                raise SliceShapeError(
                    f"{required!r} bundle is missing leaf {leaf_key!r}"
                )
            _validate_leaf(f"{required}.{leaf_key}", bundle[leaf_key])

    if split_policy == "no_split":
        raise SliceShapeError(
            "split_policy='no_split' is reserved for reference-shape outputs; "
            "B-shape requires a real split policy"
        )

    train_bundle = {
        "features": loader_output["train"]["features"],
        "ground_truth": loader_output["train"]["ground_truth"],
    }
    test_bundle_for_agent = {
        "features": loader_output["test"]["features"],
        # ground_truth is intentionally omitted from the agent payload
    }
    held_out = loader_output["test"]["ground_truth"]

    agent_payload = {
        "train": train_bundle,
        "test": test_bundle_for_agent,
    }
    return SliceResult(
        shape="b_shape",
        agent_payload=agent_payload,
        held_out=held_out,
        split_policy=split_policy,
        split_metadata=dict(split_metadata) if split_metadata else None,
    )


# HDF5 writers


def write_dataset_h5(path: Path, result: SliceResult) -> None:
    """Serialize ``result.agent_payload`` to ``path`` as HDF5.

    The on-disk hierarchy mirrors the dict hierarchy 1:1: each ndarray
    leaf becomes an HDF5 dataset, each ``{sym: ndarray}`` panel becomes
    an HDF5 group with one dataset per symbol. Top-level metadata
    (``split_policy``, ``split_metadata``, ``shape``) is written as
    HDF5 attributes on the root group so the runtime loader can
    reconstruct any context the agent might want to inspect.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        if not isinstance(result.agent_payload, Mapping):
            raise SliceShapeError(
                "agent_payload must be a mapping at the top level"
            )
        _write_into_group(f, result.agent_payload)
        f.attrs["shape"] = result.shape
        f.attrs["split_policy"] = result.split_policy
        if result.split_metadata:
            f.attrs["split_metadata_json"] = _json_dumps_compact(result.split_metadata)


def write_test_gt_h5(path: Path, held_out: Any) -> None:
    """Serialize the held-out target to ``path`` as HDF5.

    ``held_out`` is either an ndarray (single-symbol case) or a
    ``{sym: ndarray}`` mapping (panel case). The runtime evaluator's
    ``_load_reference_data`` reads this file and returns the
    reconstructed object directly to the score loop. The contents are
    stored under ``/ground_truth`` (group for panel, dataset for single).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        gt_group = f.create_group("ground_truth")
        if isinstance(held_out, np.ndarray):
            # Store the ndarray as a single dataset directly under the group
            # using a sentinel name; readers iterate the group transparently.
            # Simpler: replace the group with a top-level dataset called
            # "ground_truth" instead. But to keep one shape on disk, we use
            # a group containing a single "_value" dataset for ndarrays and
            # one dataset per symbol for panels. The reader normalises.
            gt_group.create_dataset(
                "_value", data=held_out, compression="gzip", shuffle=True
            )
        elif isinstance(held_out, Mapping):
            for sym, arr in held_out.items():
                if not isinstance(arr, np.ndarray):
                    raise SliceShapeError(
                        f"held_out[{sym!r}] must be ndarray; got {type(arr).__name__}"
                    )
                gt_group.create_dataset(
                    sym, data=arr, compression="gzip", shuffle=True
                )
        else:
            raise SliceShapeError(
                f"held_out must be ndarray or {{sym: ndarray}} dict; got "
                f"{type(held_out).__name__}"
            )


# Internal helpers


def _validate_leaf(label: str, value: Any) -> None:
    """A leaf in the B-shape dict is an ndarray or a {str: ndarray} mapping."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise SliceShapeError(f"{label}: ndarray is empty")
        return
    if isinstance(value, Mapping):
        if not value:
            raise SliceShapeError(f"{label}: panel dict is empty")
        for sym, arr in value.items():
            if not isinstance(sym, str):
                raise SliceShapeError(
                    f"{label}: panel keys must be str; got {type(sym).__name__}"
                )
            if not isinstance(arr, np.ndarray):
                raise SliceShapeError(
                    f"{label}[{sym!r}]: panel value must be np.ndarray; got "
                    f"{type(arr).__name__}"
                )
            if arr.size == 0:
                raise SliceShapeError(f"{label}[{sym!r}]: ndarray is empty")
        return
    raise SliceShapeError(
        f"{label}: must be np.ndarray or {{str: np.ndarray}} dict; got "
        f"{type(value).__name__}"
    )


def _write_into_group(group: h5py.Group, obj: Mapping[str, Any]) -> None:
    """Write each key of ``obj`` directly under ``group``.

    Each ndarray child becomes a dataset named after the key. Each
    Mapping child becomes a sub-group recursed into. Paths are kept
    RELATIVE to the parent group (no leading ``"/"``) — passing
    absolute paths to ``require_group`` would resolve them at file
    root and produce an incorrectly flat layout.
    """
    for key, child in obj.items():
        if not isinstance(key, str):
            raise SliceShapeError(
                f"all dict keys must be str; got {type(key).__name__} ({key!r})"
            )
        if isinstance(child, np.ndarray):
            group.create_dataset(
                key, data=child, compression="gzip", shuffle=True
            )
        elif isinstance(child, Mapping):
            sub = group.create_group(key)
            _write_into_group(sub, child)
        else:
            raise SliceShapeError(
                f"unsupported type {type(child).__name__} at {group.name}/{key}"
            )


def _json_dumps_compact(obj: Mapping[str, Any]) -> str:
    """Stable JSON encoding for HDF5 attribute storage."""
    import json

    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
