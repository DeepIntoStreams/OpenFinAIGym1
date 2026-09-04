"""Per-task ``load.py`` generation for Phase 4 task construction.

The loader's target/horizon/leakage decisions are task-level, so loader
codegen runs inside ``construct_tasks`` (with full paper + summary +
experiments + rewards context) rather than in Phase 2 alongside the raw
download. The approved ``load.py`` is staged at ``staging_dir/load.py``
(sibling to the assembled ``evaluator.py``) and copied into the Harbor
task package at install time.
"""

from openfinai_pipeline.benchmark.loader.executor import (
    LOADER_REQUIRES_GT_MODELS,
    LoaderResult,
    generate_task_loader,
    run_loader_script,
)

__all__ = [
    "LOADER_REQUIRES_GT_MODELS",
    "LoaderResult",
    "generate_task_loader",
    "run_loader_script",
]
