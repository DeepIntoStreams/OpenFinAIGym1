#!/usr/bin/env python
"""Delete generated pipeline outputs; use --dry-run to preview.

The default covers all phase outputs and generated task trees while preserving
.gitkeep files and root directories. Use --include or --exclude to narrow it.
"""
import argparse
import os
import shutil
import stat
import sys
from pathlib import Path


def _force_rmtree(path: Path) -> None:
    """Remove a tree containing Windows read-only files."""

    def _on_error(func, target, exc):  # signature: 3.12 onexc / 3.x onerror
        try:
            os.chmod(target, stat.S_IWRITE)
        except OSError:
            pass
        try:
            func(target)
        except OSError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_on_error)
    else:
        shutil.rmtree(path, onerror=lambda f, p, _e: _on_error(f, p, None))

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Each phase target -> the directories whose contents should be removed.
# Order within a target is preserved in the printed output.
TARGETS: dict[str, list[str]] = {
    "papers": [
        "data/pipeline_output/papers",
    ],
    "datasets": [
        "data/pipeline_output/datasets",
        "data/pipeline_output/.staging/datasets",
    ],
    "rewards": [
        "data/pipeline_output/rewards",
        "data/pipeline_output/.staging/rewards",
    ],
    "tasks": [
        "data/pipeline_output/.staging/tasks",
        "tasks/generated",
        "tasks/routed",
    ],
    "logs": [
        "data/pipeline_output/logs",
    ],
    "cache": [
        "data/pipeline_output/.cache",
    ],
}


def _resolve_targets(args: argparse.Namespace) -> list[str]:
    if args.include:
        # Preserve user-specified order, drop duplicates.
        seen: set[str] = set()
        ordered: list[str] = []
        for name in args.include:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered
    if args.exclude:
        excluded = set(args.exclude)
        return [t for t in TARGETS if t not in excluded]
    return list(TARGETS)


def _count_files(path: Path) -> int:
    return sum(
        1 for p in path.rglob("*") if p.is_file() and p.name != ".gitkeep"
    )


def _clean_directory(target_dir: Path, dry_run: bool) -> tuple[int, int]:
    """Remove every child of ``target_dir`` except ``.gitkeep``.

    Returns ``(files_seen, files_removed)``. In dry-run, ``files_removed``
    is always 0.
    """
    if not target_dir.exists():
        return (0, 0)

    files_seen = 0
    files_removed = 0

    for item in sorted(target_dir.iterdir()):
        if item.name == ".gitkeep":
            continue
        rel = item.relative_to(PROJECT_ROOT)
        if item.is_dir():
            n = _count_files(item)
            files_seen += n
            if dry_run:
                print(f"  would remove dir:  {rel}  ({n} files)")
            else:
                _force_rmtree(item)
                files_removed += n
                print(f"  removed dir:  {rel}  ({n} files)")
        else:
            files_seen += 1
            if dry_run:
                print(f"  would remove file: {rel}")
            else:
                item.unlink()
                files_removed += 1
                print(f"  removed file: {rel}")

    return (files_seen, files_removed)


def _print_targets() -> None:
    print("Available targets:")
    width = max(len(name) for name in TARGETS)
    for name in TARGETS:
        paths = TARGETS[name]
        print(f"  {name:<{width}}  {paths[0]}")
        for extra in paths[1:]:
            print(f"  {'':<{width}}  {extra}")


def main() -> None:
    target_choices = sorted(TARGETS.keys())
    parser = argparse.ArgumentParser(
        description="Clean pipeline outputs from previous runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Targets:\n  "
            + ", ".join(target_choices)
            + "\n\nRun with --list-targets to see the directories each one cleans."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be removed without deleting anything.",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="Print the available targets (and the dirs they clean) and exit.",
    )
    select_group = parser.add_mutually_exclusive_group()
    select_group.add_argument(
        "--include",
        nargs="+",
        choices=target_choices,
        metavar="TARGET",
        help=(
            "Clean only these pipeline phase outputs. "
            f"Choices: {', '.join(target_choices)}. Default: clean all."
        ),
    )
    select_group.add_argument(
        "--exclude",
        nargs="+",
        choices=target_choices,
        metavar="TARGET",
        help=(
            "Clean every target except these. "
            f"Choices: {', '.join(target_choices)}. Cannot combine with --include."
        ),
    )
    args = parser.parse_args()

    if args.list_targets:
        _print_targets()
        return

    selected = _resolve_targets(args)
    if not selected:
        print("No targets selected; nothing to clean.")
        return

    print(f"Cleaning targets: {', '.join(selected)}")
    if args.dry_run:
        print("(dry-run mode -- no files will be deleted)")

    total_seen = 0
    total_removed = 0
    cleaned_dirs: list[Path] = []

    for name in selected:
        for rel in TARGETS[name]:
            target_dir = PROJECT_ROOT / rel
            if not target_dir.exists():
                print(f"\n[{name}] {rel}: does not exist, skipping")
                continue
            print(f"\n[{name}] {rel}:")
            seen, removed = _clean_directory(target_dir, args.dry_run)
            total_seen += seen
            total_removed += removed
            cleaned_dirs.append(target_dir)
            if seen == 0:
                print("  (already clean)")

    if not args.dry_run:
        for d in cleaned_dirs:
            d.mkdir(parents=True, exist_ok=True)
        suffix = "y" if len(cleaned_dirs) == 1 else "ies"
        print(
            f"\nDone. Removed {total_removed} file(s) "
            f"across {len(cleaned_dirs)} director{suffix}."
        )
    else:
        suffix = "y" if len(cleaned_dirs) == 1 else "ies"
        print(
            f"\nDry-run complete. Would remove {total_seen} file(s) "
            f"across {len(cleaned_dirs)} director{suffix}."
        )


if __name__ == "__main__":
    main()
