"""Install-time helpers used by ``benchmark/builders/task_builder.py``.

These modules turn the LLM-generated ``load.py`` plus the assembled
evaluator into the on-disk task package shape that Harbor consumes:

* :mod:`openfinai_pipeline.benchmark.install.slicer` — invokes the
  staged loader once, applies the chronological train/test split, and
  serializes the bundles to HDF5 (``dataset.h5`` for the agent;
  ``test_ground_truth.h5`` for the verifier).
* :mod:`openfinai_pipeline.benchmark.install.reward_bundler` — AST-walks
  the curated reward bank plus any auto-generated reward modules to
  produce a single self-contained ``task_rewards.py`` containing only
  the classes the assembled evaluator imports.
* :mod:`openfinai_pipeline.benchmark.install.runtime_loader` — exposes
  the source of the trivial runtime ``load.py`` that replaces the LLM
  loader inside the installed task package; the runtime loader simply
  reads ``dataset.h5`` and substitutes ``None`` for the held-out test
  ground_truth.
"""

from openfinai_pipeline.benchmark.install.runtime_loader import (
    TRIVIAL_RUNTIME_LOADER_SRC,
    read_dataset_h5,
)
from openfinai_pipeline.benchmark.install.slicer import (
    SliceResult,
    slice_b_shape,
    write_dataset_h5,
    write_test_gt_h5,
)

__all__ = [
    "SliceResult",
    "TRIVIAL_RUNTIME_LOADER_SRC",
    "read_dataset_h5",
    "slice_b_shape",
    "write_dataset_h5",
    "write_test_gt_h5",
]
