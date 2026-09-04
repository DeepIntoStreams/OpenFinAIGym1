#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Harbor registry JSON from task globs.")
    parser.add_argument("--tasks-root", default="tasks", help="Task root directory.")
    parser.add_argument("--task-glob", action="append", required=True, help="Glob of task directories to include.")
    parser.add_argument("--dataset-name", required=True, help="Registry dataset name.")
    parser.add_argument("--dataset-version", default="1.0", help="Registry dataset version.")
    parser.add_argument("--description", default="Generated Harbor registry subset.", help="Registry description.")
    parser.add_argument("--output-path", required=True, help="Destination JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks_root = Path(args.tasks_root)
    matched: dict[str, Path] = {}

    for task_glob in args.task_glob:
        for path in sorted(tasks_root.glob(task_glob)):
            if path.is_dir():
                matched[path.name] = path

    if not matched:
        raise RuntimeError(f"No tasks matched globs={args.task_glob!r} under {tasks_root}")

    payload = [
        {
            "name": args.dataset_name,
            "version": args.dataset_version,
            "description": args.description,
            "tasks": [{"name": name, "path": str(path)} for name, path in sorted(matched.items())],
            "metrics": [{"type": "mean"}],
        }
    ]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[build-task-registry] dataset={args.dataset_name}@{args.dataset_version} "
        f"tasks={len(matched)} output={output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
