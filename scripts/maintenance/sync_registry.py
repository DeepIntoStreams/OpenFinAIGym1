#!/usr/bin/env python
"""
Scan tasks/ and regenerate registry.json with all valid tasks.

A "valid task" is any direct subdirectory of tasks/ that contains a task.toml
file and does NOT start with underscore (so tasks/_template is skipped).

Run this whenever you add or remove a task:
    python scripts/maintenance/sync_registry.py

It is idempotent — running it multiple times produces the same output.
"""

from __future__ import annotations

import json
from pathlib import Path


REGISTRY_NAME = "openfinai"
REGISTRY_VERSION = "1.0"
REGISTRY_DESCRIPTION = "OpenFinGym — financial agent evaluation benchmark"


def scan_tasks(tasks_dir: Path) -> list[dict]:
    """Return a sorted list of {name, path} dicts for all valid tasks."""
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

    tasks = []
    for entry in sorted(tasks_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_"):
            continue  # skip _template, etc.
        if not (entry / "task.toml").exists():
            continue  # not a valid task
        tasks.append({
            "name": entry.name,
            "path": f"{tasks_dir.name}/{entry.name}",
        })
    return tasks


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    tasks_dir = project_root / "tasks"
    registry_path = project_root / "registry.json"

    tasks = scan_tasks(tasks_dir)
    if not tasks:
        print(f"[sync-registry] WARNING: no valid tasks found in {tasks_dir}")

    registry = [{
        "name": REGISTRY_NAME,
        "version": REGISTRY_VERSION,
        "description": REGISTRY_DESCRIPTION,
        "tasks": tasks,
        "metrics": [{"type": "mean"}],
    }]

    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    print(f"[sync-registry] Registered {len(tasks)} tasks in {registry_path.name}:")
    for t in tasks:
        print(f"  - {t['name']}  ({t['path']})")


if __name__ == "__main__":
    main()
