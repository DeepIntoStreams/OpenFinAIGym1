"""Runtime loader for the curated offline forecasting bundle.

Reads the per-symbol ``dataset.h5`` the curated verifier provisions into
``/data`` and reconstructs the
``{symbol: {X_train, y_train, X_test, reference_train, reference_test}}``
dict the bundled task shim's accessors expect.

The held-out test target (``y_test``) is intentionally absent from this
file — it lives only in the verifier-side
``environment/eval-data/test_ground_truth.h5`` mount, which the agent
container never sees. The shim's ``get_ground_truth()`` therefore raises
``PermissionError`` (inherited from ``ForecastingTask``).

Self-contained on purpose (stdlib + h5py + numpy): imported from the
installed bundle, never from the pipeline package. Keep in sync with the
curated forecasting handler's ``dataset.h5`` layout (see
``openfinai_harbor.verifier.curated.handlers.forecasting``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np


def load(data_dir) -> Dict[str, Any]:
    """Read ``dataset.h5`` → ``{symbol: {array_name: ndarray}}``."""
    data_dir = Path(data_dir)
    h5_path = data_dir / "dataset.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"curated forecasting loader expected dataset.h5 at {h5_path!s}; "
            "the bundle is provisioned by the curated verifier at spawn time."
        )
    out: Dict[str, Any] = {}
    with h5py.File(h5_path, "r") as f:
        layout = str(f.attrs.get("layout", "per_symbol"))
        if layout == "per_symbol":
            for sym in f.keys():
                grp = f[sym]
                out[sym] = {k: np.asarray(grp[k][...]) for k in grp.keys()}
        else:
            # single-output layout: expose under a sentinel key so the
            # shim's per-symbol accessors stay uniform.
            out["_"] = {k: np.asarray(f[k][...]) for k in f.keys()}
    return out
