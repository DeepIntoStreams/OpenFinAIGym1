"""Trivial runtime ``load.py`` source + a matching read helper.

After install, the LLM-generated ``load.py`` is replaced by the source
in :data:`TRIVIAL_RUNTIME_LOADER_SRC` — a tiny module that reads
``data_dir/dataset.h5`` and reconstructs the B-shape (or reference-shape)
dict the framework expects.

The ``test["ground_truth"]`` leaf is intentionally absent from the on-disk
HDF5 (the install slicer omits it so the agent cannot read it). The
runtime loader substitutes ``None`` for it when reconstructing the dict
so downstream code can detect that the test target is held out and pull
it from ``/eval-data/test_ground_truth.h5`` instead.

:func:`read_dataset_h5` is the matching reader exposed for tests and the
dry-fire validator; the runtime loader uses an inlined copy (it cannot
import from the pipeline package at runtime — the installed Harbor
container does not have the pipeline source on its path).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

__all__ = [
    "TRIVIAL_RUNTIME_LOADER_SRC",
    "read_dataset_h5",
    "read_test_gt_h5",
]


# Read helpers (used by tests + dry-fire; runtime loader has its own copy)


def read_dataset_h5(path: Path) -> dict[str, Any]:
    """Reconstruct the B-shape (or reference-shape) dict from ``dataset.h5``.

    Always substitutes ``None`` for ``test["ground_truth"]`` — that leaf
    is intentionally not in the file (held out from the agent). For
    reference-shape, returns ``{"reference": ...}`` unchanged.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        out = _read_group(f)
        if "test" in out and isinstance(out["test"], dict):
            out["test"].setdefault("ground_truth", None)
        # Surface the metadata attrs so callers can inspect the policy.
        if "split_policy" in f.attrs:
            out["split_policy"] = str(f.attrs["split_policy"])
        if "split_metadata_json" in f.attrs:
            import json

            try:
                out["split_metadata"] = json.loads(str(f.attrs["split_metadata_json"]))
            except (ValueError, TypeError):
                out["split_metadata"] = None
        else:
            out["split_metadata"] = None
    return out


def read_test_gt_h5(path: Path) -> Any:
    """Read the held-out test ground_truth from ``test_ground_truth.h5``.

    Returns either an ndarray (single-symbol) or a ``{sym: ndarray}`` dict
    (panel case). The on-disk shape is always a ``/ground_truth`` group;
    a single-ndarray held-out is stored under the sentinel key
    ``_value`` and unwrapped here, while panel data is stored as one
    dataset per symbol and returned as a dict.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "ground_truth" not in f:
            raise KeyError(
                f"{path}: missing required '/ground_truth' group"
            )
        gt = f["ground_truth"]
        if not isinstance(gt, h5py.Group):
            raise TypeError(
                f"{path}: '/ground_truth' must be a group; got "
                f"{type(gt).__name__}"
            )
        keys = list(gt.keys())
        if keys == ["_value"]:
            return np.asarray(gt["_value"][...])
        # Panel case: one dataset per symbol.
        out: dict[str, Any] = {}
        for key in keys:
            item = gt[key]
            if not isinstance(item, h5py.Dataset):
                raise TypeError(
                    f"{path}: '/ground_truth/{key}' must be a dataset; got "
                    f"{type(item).__name__}"
                )
            out[key] = np.asarray(item[...])
        return out


def _read_group(g: h5py.Group) -> dict[str, Any]:
    """Recursive walk: HDF5 group → nested dict of ndarrays."""
    out: dict[str, Any] = {}
    for key in g.keys():
        item = g[key]
        out[key] = _read_group_or_dataset(item)
    return out


def _read_group_or_dataset(item: Any) -> Any:
    if isinstance(item, h5py.Dataset):
        return np.asarray(item[...])
    if isinstance(item, h5py.Group):
        return _read_group(item)
    raise TypeError(f"unexpected HDF5 node type {type(item).__name__}")


# Runtime ``load.py`` source (written to ``/data/load.py`` at install time)


# IMPORTANT: this string is the literal source of the runtime loader.
# Keep it self-contained — it will be imported from the installed task
# package, NOT from the pipeline. No imports from openfinai_pipeline.
TRIVIAL_RUNTIME_LOADER_SRC: str = '''\
"""Auto-generated runtime loader for an installed OpenFinGym task.

Reads ``data_dir/dataset.h5`` and reconstructs the B-shape (or
reference-shape) dict the framework's ``BaseTask.load_data`` expects.
``test["ground_truth"]`` is omitted from the file by design (held out
from the agent) and substituted with ``None`` here; the verifier's
evaluator pulls the real test ground_truth from a separate
``/eval-data/test_ground_truth.h5`` artifact the agent cannot read.

Do NOT edit this file by hand — it is overwritten on every install.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def load(data_dir) -> dict:
    """Read the task's pre-sliced HDF5 bundle and return the framework dict."""
    data_dir = Path(data_dir)
    h5_path = data_dir / "dataset.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"runtime loader expected dataset.h5 at {h5_path!s}; "
            "the task package may have been installed without the install-time "
            "slicer. Re-install via openfinai_pipeline.benchmark."
        )
    with h5py.File(h5_path, "r") as f:
        out = _read_group(f)
        if "test" in out and isinstance(out["test"], dict):
            out["test"].setdefault("ground_truth", None)
        if "split_policy" in f.attrs:
            out["split_policy"] = str(f.attrs["split_policy"])
        if "split_metadata_json" in f.attrs:
            try:
                out["split_metadata"] = json.loads(str(f.attrs["split_metadata_json"]))
            except (ValueError, TypeError):
                out["split_metadata"] = None
        else:
            out["split_metadata"] = None
    return out


def _read_group(g) -> dict:
    out: dict[str, Any] = {}
    for key in g.keys():
        out[key] = _read_node(g[key])
    return out


def _read_node(item):
    if isinstance(item, h5py.Dataset):
        return np.asarray(item[...])
    if isinstance(item, h5py.Group):
        return _read_group(item)
    raise TypeError(f"unexpected HDF5 node type {type(item).__name__}")
'''
