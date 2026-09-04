#!/usr/bin/env python
"""Pack generated task bundles and optional pipeline output into a zip.

Task bundles are self-contained. ``--include-pipeline-output`` adds published
phase artifacts; ``--include-internals`` also adds caches, staging, and logs.
Internals may contain prompts, model output, downloaded data, or operational
details and must be inspected before sharing.
"""

import argparse
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PIPELINE_OUTPUT_ROOT = "data/pipeline_output"

PIPELINE_OUTPUT_EXCLUDE: set[str] = {".cache", ".staging", "logs"}

TASK_INCLUDE_DIRS: list[str] = [
    "tasks/generated",
    "tasks/routed",
]

SKIP_NAMES: set[str] = {".gitkeep", "__pycache__", ".DS_Store", "Thumbs.db"}

SKIP_EXTENSIONS: set[str] = {".pyc", ".pyo", ".pyd"}


# Helpers
def _should_skip(path: Path) -> bool:
    """Return True if *path* should be excluded from the archive."""
    if path.name in SKIP_NAMES:
        return True
    if any(part in SKIP_NAMES for part in path.parts):
        return True
    if path.suffix in SKIP_EXTENSIONS:
        return True
    return False


def _resolve_include_dirs(
    *,
    include_pipeline_output: bool,
    include_internals: bool,
) -> list[Path]:
    """Compute concrete directories to walk.

    Tasks (``tasks/generated``, ``tasks/routed``) are always included.

    When *include_pipeline_output* is True, the children of
    ``data/pipeline_output`` are added, minus ``.cache``/``.staging``/``logs``.

    When *include_internals* is True, those three excluded children are
    added too. ``include_internals`` implies ``include_pipeline_output``.
    """
    dirs: list[Path] = []

    for rel in TASK_INCLUDE_DIRS:
        p = PROJECT_ROOT / rel
        if p.exists():
            dirs.append(p)

    if include_internals or include_pipeline_output:
        excluded = set() if include_internals else PIPELINE_OUTPUT_EXCLUDE
        pipeline_root = PROJECT_ROOT / PIPELINE_OUTPUT_ROOT
        if pipeline_root.exists():
            for child in sorted(pipeline_root.iterdir()):
                if child.is_dir() and child.name not in excluded:
                    dirs.append(child)
                elif child.is_file() and not _should_skip(child):
                    # Loose files directly under pipeline_output (rare, but keep them).
                    dirs.append(child)

    return dirs


def _collect_files(
    *,
    include_pipeline_output: bool,
    include_internals: bool,
) -> list[Path]:
    """Walk include roots and return a sorted list of files to pack."""
    files: list[Path] = []
    roots = _resolve_include_dirs(
        include_pipeline_output=include_pipeline_output,
        include_internals=include_internals,
    )
    for root in roots:
        if root.is_file():
            if not _should_skip(root):
                files.append(root)
            continue
        for file_path in sorted(root.rglob("*")):
            if file_path.is_file() and not _should_skip(file_path):
                files.append(file_path)
    return files


def _human_size(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


# Main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack pipeline-generated files into a shareable zip.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output zip file name (default: generated_<timestamp>.zip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be zipped without creating the archive.",
    )
    parser.add_argument(
        "--include-pipeline-output",
        action="store_true",
        help=(
            "Also pack data/pipeline_output/ (papers, datasets, rewards, "
            "manifests, ...) minus .cache/.staging/logs. Off by default — "
            "installed tasks already carry their own task_rewards.py so "
            "data/pipeline_output/rewards/ is not needed at task runtime."
        ),
    )
    parser.add_argument(
        "--include-internals",
        action="store_true",
        help=(
            "Also pack data/pipeline_output/.cache, .staging and logs "
            "(implies --include-pipeline-output; use for a full debug bundle)."
        ),
    )
    args = parser.parse_args()

    files = _collect_files(
        include_pipeline_output=args.include_pipeline_output,
        include_internals=args.include_internals,
    )

    if not files:
        print("No generated files found to pack.")
        return

    if args.dry_run:
        total = 0
        print(f"Would pack {len(files)} file(s):\n")
        for f in files:
            size = f.stat().st_size
            total += size
            print(f"  {f.relative_to(PROJECT_ROOT).as_posix()}  ({_human_size(size)})")
        print(f"\nTotal uncompressed: {_human_size(total)}")
        return

    output_name = args.output or f"generated_{datetime.now():%Y%m%d_%H%M%S}.zip"
    output_path = PROJECT_ROOT / output_name

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arcname = f.relative_to(PROJECT_ROOT).as_posix()
            zf.write(f, arcname)

    print(
        f"Packed {len(files)} file(s) -> {output_path.name} "
        f"({_human_size(output_path.stat().st_size)})"
    )


if __name__ == "__main__":
    main()
