"""Deterministic, checked-in test files staged into Phase 4 task packages.

Replaces the LLM-authored ``test_task.py``. The files in this package are
copied verbatim into ``staging_dir/`` by ``write_deterministic_harbor_assets``
during task construction. Each task carries a per-task ``manifest.json``
sidecar that supplies the task-specific assertion values (class name,
expected score-dict keys, expected agent-output filenames).
"""
