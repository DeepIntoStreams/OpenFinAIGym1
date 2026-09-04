"""SkyRL-compatible discovery of OpenFinGym task directories."""

from __future__ import annotations

from pathlib import Path
from typing import List

from loguru import logger


class OpenFinAITaskDataset:
    """Discover directories containing instruction.md and expose SkyRL rows."""

    def __init__(self, task_dirs: List[str]):
        """Construct an :class:`OpenFinAITaskDataset` from one or more roots.

        Args:
            task_dirs: List of paths to directories containing Harbor tasks.
                Each directory is scanned for valid task subdirectories.
        """
        self.task_paths: List[Path] = []

        for dir_path in task_dirs:
            source = Path(dir_path)
            if not source.exists():
                logger.warning(f"Task directory does not exist: {dir_path}")
                continue

            if self._is_valid_task(source):
                self.task_paths.append(source)
                continue

            for child in sorted(source.iterdir()):
                if child.is_dir() and self._is_valid_task(child):
                    self.task_paths.append(child)

        logger.info(f"OpenFinAITaskDataset: found {len(self.task_paths)} tasks")

    @staticmethod
    def _is_valid_task(path: Path) -> bool:
        return path.is_dir() and (path / "instruction.md").exists()

    def __getitem__(self, index: int) -> dict:
        if index >= len(self.task_paths):
            raise IndexError(f"Index {index} out of range (dataset size: {len(self.task_paths)})")
        return {
            "prompt": str(self.task_paths[index]),
            "env_class": None,
            "env_extras": {"data_source": str(self.task_paths[index])},
            "uid": str(index),
        }

    def __len__(self) -> int:
        return len(self.task_paths)

    def __iter__(self):
        for i in range(len(self.task_paths)):
            yield self[i]

    def collate_fn(self, item_list):
        return item_list
