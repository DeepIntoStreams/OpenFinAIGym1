"""Curated trading task catalog loader for Phase 1c triage.

Scans ``tasks/*/triage_descriptor.toml`` and produces structured
``CuratedTradingDescriptor`` records consumed by ``papers/triage.py``.
The descriptor file is sibling to each curated bundle's ``task.toml``
and is the single source of triage-only data — structural fields,
per-config-schema-field metadata, and LLM-prompt prose. The runtime
``task.toml`` next to it stays lean (~22 lines for trading bundles)
and is not read by triage.

The catalog is serialized into the triage LLM prompt (markdown form)
and hashed for inclusion on every triage record so catalog drift is
detectable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


DESCRIPTOR_FILENAME = "triage_descriptor.toml"


class CuratedTradingDescriptor(BaseModel):
    """Schema mirroring the per-bundle ``triage_descriptor.toml`` shape.

    Loaded from ``tasks/<bundle>/triage_descriptor.toml`` where
    ``triage_eligible == true``. The ``task_id`` and ``class_name``
    fields are pulled from the sibling ``task.toml`` (``[metadata].name``
    and ``[curated].class_name``) — they are not duplicated in the
    descriptor file.
    """

    task_id: str
    class_name: str
    # Base curated family (e.g. "offline_trading") + container image, both
    # pulled from the sibling task.toml. A routed overlay materialized from
    # this descriptor inherits them so the resulting bundle is a first-class
    # curated bundle the harbor verifier dispatches without routed-awareness.
    base_family: str
    docker_image: str | None = None
    schema_version: str = "1"
    triage_eligible: bool = True
    asset_classes: list[str] = Field(default_factory=list)
    exchanges: list[str] = Field(default_factory=list)
    mode: str
    action_space_description: str
    task_description: str
    action_format_description: str
    evaluation_description: str
    config_schema: dict[str, dict[str, Any]] = Field(default_factory=dict)
    catalog_defaults: dict[str, Any] = Field(default_factory=dict)


def load_curated_trading_catalog(tasks_root: Path) -> list[CuratedTradingDescriptor]:
    """Scan ``tasks/*/triage_descriptor.toml`` and return all triage-eligible
    descriptors.

    Sub-paths under ``tasks/routed/`` are skipped — those are derived overlays
    written by triage itself, not source curated tasks. The result is sorted by
    ``task_id`` for deterministic catalog hashing.
    """
    tasks_root = Path(tasks_root)
    descriptors: list[CuratedTradingDescriptor] = []
    for desc_path in sorted(tasks_root.glob(f"*/{DESCRIPTOR_FILENAME}")):
        # Skip the routed-overlay subtree if a future restructure puts it under tasks/.
        if desc_path.parent.name == "routed":
            continue
        descriptor = _load_descriptor_from_toml(desc_path)
        if descriptor is None:
            continue
        descriptors.append(descriptor)
    descriptors.sort(key=lambda d: d.task_id)
    return descriptors


def _load_descriptor_from_toml(desc_path: Path) -> CuratedTradingDescriptor | None:
    """Parse one ``triage_descriptor.toml``; return None if marked ineligible.

    Reads ``task_id`` and ``class_name`` from the sibling ``task.toml``
    (``[metadata].name`` / ``[curated].class_name``) so the descriptor file
    doesn't duplicate identifiers the runtime config already owns.
    """
    descriptor_block = tomllib.loads(desc_path.read_text(encoding="utf-8"))
    if not descriptor_block.get("triage_eligible", False):
        return None

    task_toml_path = desc_path.parent / "task.toml"
    if not task_toml_path.exists():
        raise ValueError(
            f"{desc_path}: sibling task.toml not found at {task_toml_path} — "
            f"triage descriptors must live next to a curated task's task.toml"
        )
    task_raw = tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
    task_id = (task_raw.get("metadata") or {}).get("name")
    class_name = (task_raw.get("curated") or {}).get("class_name")
    base_family = (task_raw.get("metadata") or {}).get("family")
    docker_image = (task_raw.get("environment") or {}).get("docker_image")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(
            f"{task_toml_path}: missing [metadata].name (required for "
            f"triage descriptor at {desc_path})"
        )
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError(
            f"{task_toml_path}: missing [curated].class_name (required for "
            f"triage descriptor at {desc_path})"
        )
    if not isinstance(base_family, str) or not base_family.strip():
        raise ValueError(
            f"{task_toml_path}: missing [metadata].family (required so routed "
            f"overlays inherit a harbor-dispatchable curated family)"
        )

    payload = {
        **descriptor_block,
        "task_id": task_id,
        "class_name": class_name,
        "base_family": base_family,
        "docker_image": docker_image
        if isinstance(docker_image, str) and docker_image.strip()
        else None,
    }
    return CuratedTradingDescriptor.model_validate(payload)


def compute_catalog_hash(catalog: list[CuratedTradingDescriptor]) -> str:
    """sha256 of canonical JSON of the catalog — for stamping triage records."""
    canonical = json.dumps(
        [d.model_dump(mode="json") for d in catalog],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def serialize_catalog_for_prompt(catalog: list[CuratedTradingDescriptor]) -> str:
    """Render the catalog as markdown for inclusion in the triage LLM prompt."""
    if not catalog:
        return "(catalog is empty — no curated trading tasks available)"
    lines: list[str] = []
    for d in catalog:
        lines.append(f"### `{d.task_id}` (class `{d.class_name}`)")
        lines.append(f"- **Asset classes**: {', '.join(d.asset_classes) or '(unspecified)'}")
        lines.append(f"- **Exchanges**: {', '.join(d.exchanges) or '(unspecified)'}")
        lines.append(f"- **Mode**: {d.mode}")
        lines.append(f"- **Action space**: {d.action_space_description}")
        lines.append("- **Description**:")
        for para in d.task_description.strip().split("\n"):
            lines.append(f"  > {para}")
        lines.append("- **Config schema** (fields you can populate in `task_config`):")
        for field_name, field_spec in d.config_schema.items():
            lines.extend(_render_field_spec(field_name, field_spec))
        lines.append(
            "- **Catalog defaults** — placeholder values shown for reference. "
            "These fill in ONLY for fields the paper genuinely never addresses. "
            "They are NOT preferred answers; override them in `task_config` "
            "whenever the paper specifies anything different (asset universe, "
            "dates, bar resolution, episode length, sizing, etc.):"
        )
        lines.append("  ```json")
        lines.append("  " + json.dumps(d.catalog_defaults, indent=2, ensure_ascii=False).replace("\n", "\n  "))
        lines.append("  ```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_field_spec(field_name: str, spec: dict[str, Any]) -> list[str]:
    """Render one config-schema entry as markdown bullets."""
    type_str = spec.get("type", "?")
    required = spec.get("required", False)
    head = f"  - `{field_name}` ({type_str})"
    if required:
        head += " — **required**"
    elif "default" in spec:
        head += f" — default `{spec['default']!r}`"
    out = [head]
    notes = spec.get("notes")
    if notes:
        out.append(f"    - {notes}")
    examples = spec.get("examples")
    if examples:
        out.append(f"    - Examples: {', '.join(repr(e) for e in examples)}")
    interval_options = spec.get("interval_options")
    if interval_options:
        out.append(f"    - Interval options: {', '.join(interval_options)}")
    fmt = spec.get("format")
    if fmt:
        out.append(f"    - Format: `{fmt}`")
    return out


def find_descriptor(
    catalog: list[CuratedTradingDescriptor], task_id: str
) -> CuratedTradingDescriptor | None:
    """O(n) lookup; the catalog is small (≤ a handful of curated trading tasks)."""
    for d in catalog:
        if d.task_id == task_id:
            return d
    return None
