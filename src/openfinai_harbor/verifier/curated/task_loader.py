"""Load non-package curated task modules from task directories.

Hashed module names avoid collisions without modifying ``sys.path``. The
verifier resolves the class and configuration from ``task.toml`` at startup.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover — pyproject.toml requires 3.12+
    import tomli as tomllib  # type: ignore[import-not-found]


_CURATED_FAMILIES = (
    "offline_forecasting",
    "offline_trading",
    "realtime_forecasting",
    "realtime_trading",
    "realtime_polymarket",
)


@dataclass
class CuratedTaskSpec:
    """Resolved description of a curated task bundle.

    All fields are derived from ``task.toml`` + the bundle's filesystem
    layout. The actual task class isn't imported here — the verifier
    does that at startup so import failures show up at startup time, not
    on first submission.
    """

    task_dir: Path
    task_id: str
    family: str
    class_name: str
    default_config: Dict[str, Any]
    task_module_path: Path  # tasks/<task>/task.py
    raw_toml: Dict[str, Any]


def is_curated_bundle(task_dir: Path) -> bool:
    """Return True iff ``task_dir`` looks like a curated bundle.

    Curated bundles have ``task.toml`` with a recognized ``[metadata]
    .family``. Auto-pipe bundles have ``manifest.json`` and don't set a
    family — the dispatch in ``__main__.py`` falls through to the
    auto-pipe path on this check returning False.
    """
    toml_path = task_dir / "task.toml"
    if not toml_path.exists():
        return False
    try:
        data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    family = (data.get("metadata") or {}).get("family")
    return family in _CURATED_FAMILIES


def load_curated_spec(task_dir: Path) -> CuratedTaskSpec:
    """Parse ``task.toml`` and return the dispatch spec.

    Raises ``RuntimeError`` with a precise diagnostic on missing fields
    so the operator sees ``"task.toml missing [curated].class_name"``
    rather than a KeyError stack trace.
    """
    toml_path = task_dir / "task.toml"
    if not toml_path.exists():
        raise RuntimeError(
            f"curated task.toml not found at {toml_path}; "
            "see tasks/_template/task.toml for the expected layout"
        )
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    metadata = raw.get("metadata") or {}
    curated = raw.get("curated") or {}

    family = metadata.get("family")
    if family not in _CURATED_FAMILIES:
        raise RuntimeError(
            f"task.toml [metadata].family={family!r} is not one of "
            f"{_CURATED_FAMILIES}"
        )
    name = metadata.get("name") or task_dir.name
    class_name = curated.get("class_name")
    if not class_name:
        raise RuntimeError(
            f"task.toml [curated].class_name missing for {task_dir}; "
            "set it to the curated task subclass (e.g. 'OfflineCryptoForecasting')"
        )
    default_config = dict(curated.get("default_config") or {})

    task_module_path = task_dir / "task.py"
    if not task_module_path.exists():
        raise RuntimeError(
            f"curated task.py not found at {task_module_path}; "
            "the bundle layout requires task.py at the bundle root"
        )

    return CuratedTaskSpec(
        task_dir=task_dir.resolve(),
        task_id=name,
        family=family,
        class_name=class_name,
        default_config=default_config,
        task_module_path=task_module_path.resolve(),
        raw_toml=raw,
    )


def import_task_class(spec: CuratedTaskSpec) -> type:
    """Import ``task.py`` and return the configured class.

    Uses a hashed module name so multiple tasks loaded in the same
    process don't collide on ``sys.modules``. The hash includes the
    absolute module path so reloads of the same task return a fresh
    module object (avoids stale class objects after edits).
    """
    module_name = f"_curated_task_{abs(hash(str(spec.task_module_path)))}"
    sys.modules.pop(module_name, None)  # ensure a fresh import each spawn

    s = importlib.util.spec_from_file_location(
        module_name, str(spec.task_module_path)
    )
    if s is None or s.loader is None:
        raise RuntimeError(
            f"could not build import spec for {spec.task_module_path}"
        )
    module = importlib.util.module_from_spec(s)
    sys.modules[module_name] = module
    s.loader.exec_module(module)

    cls = getattr(module, spec.class_name, None)
    if cls is None or not isinstance(cls, type):
        raise RuntimeError(
            f"class {spec.class_name!r} not found in "
            f"{spec.task_module_path} — fix task.toml's "
            "[curated].class_name to match the actual class symbol"
        )
    return cls


def instantiate_task(
    cls: type,
    *,
    config_override: Optional[Dict[str, Any]] = None,
    default_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Construct a curated task instance with merged config.

    ``config_override`` (typically per-trial overrides from the agent's
    submission) takes precedence over ``default_config`` (from
    ``task.toml [curated].default_config``). Curated tasks always accept
    ``config: dict`` as their first positional argument by convention.
    """
    base = dict(default_config or {})
    if config_override:
        base.update(config_override)
    return cls(config=base)
