import json
import re
import unicodedata
from pathlib import Path
from typing import Any

GENERATED_DATASET_CATALOG_NAME = "generated_datasets.json"


def slugify(name: str) -> str:
    normalized = (
        unicodedata.normalize("NFKD", name or "")
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower()).strip("_")
    return clean or "dataset"


def load_dataset_catalog(catalog_path: Path) -> list[dict[str, Any]]:
    if not catalog_path.exists():
        return []
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [
        _normalize_dataset_catalog_entry(item)
        for item in payload
        if isinstance(item, dict)
    ]


def load_reward_catalog(catalog_path: Path) -> list[dict[str, Any]]:
    if not catalog_path.exists():
        return []
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [
        _normalize_reward_catalog_entry(item)
        for item in payload
        if isinstance(item, dict)
    ]


def upsert_dataset_catalog_entry(
    catalog: list[dict[str, Any]],
    *,
    name: str,
    description: str,
    downloaded_dataset_description: str,
    preview: list[dict[str, Any]] | dict[str, Any] | None,
    has_ground_truth: bool = False,
    scope_ids: list[str] | None = None,
) -> bool:
    normalized_entry: dict[str, Any] = {
        "name": str(name).strip(),
        "description": description.strip(),
        "downloaded_dataset_description": str(downloaded_dataset_description).strip(),
        "preview": _normalize_preview(preview or []),
        "has_ground_truth": bool(has_ground_truth),
    }
    if not normalized_entry["name"]:
        return False
    new_scope_ids = _normalize_scope_ids(scope_ids)
    for item in catalog:
        item_name = str(item.get("name", "")).strip()
        if item_name != normalized_entry["name"]:
            continue
        # Union (never narrow) prior scope_ids: dataset relevant to {A,B}
        # stays {A,B} even when B's run re-emits without remembering A.
        merged_scope_ids = _normalize_scope_ids(
            list(item.get("scope_ids", [])) + list(new_scope_ids)
        )
        if merged_scope_ids:
            normalized_entry["scope_ids"] = merged_scope_ids
        if item != normalized_entry:
            item.clear()
            item.update(normalized_entry)
            return True
        return False
    if new_scope_ids:
        normalized_entry["scope_ids"] = list(new_scope_ids)
    catalog.append(normalized_entry)
    return True


def upsert_reward_catalog_entry(
    catalog: list[dict[str, Any]],
    reward: Any,
    *,
    spec: dict[str, Any] | None = None,
    scope_ids: list[str] | None = None,
) -> bool:
    """Insert/update a reward entry in *catalog*.

    When *spec* is supplied (the payload returned by ``write_reward_asset``)
    the entry is enriched with ``module``, ``class_name``, ``base_class``,
    and ``default_params`` so the generated catalog mirrors the curated
    ``reward_bank.json`` schema. Without *spec* the entry stays minimal
    (just ``name`` and ``description``) — used by call sites that only
    need to record the reward's existence.

    Param-name → ctx-key wiring is implicit in the source signatures
    under the canonical-name contract; no envelope is stored.
    """
    reward_name = str(getattr(reward, "name", "")).strip()
    if not reward_name:
        return False
    normalized_entry: dict[str, Any] = {
        "name": reward_name,
        "description": str(getattr(reward, "description", "")).strip(),
    }
    if isinstance(spec, dict):
        module_name = str(spec.get("module_name", "")).strip()
        class_name = str(spec.get("class_name", "")).strip()
        base_class = str(spec.get("base_class", "")).strip() or "Loss"
        if module_name:
            normalized_entry["module"] = module_name
        if class_name:
            normalized_entry["class_name"] = class_name
        normalized_entry["base_class"] = base_class
        default_params = spec.get("default_params")
        normalized_entry["default_params"] = (
            dict(default_params) if isinstance(default_params, dict) else {}
        )
    new_scope_ids = _normalize_scope_ids(scope_ids)
    for item in catalog:
        name = str(item.get("name", "")).strip()
        if name != reward_name:
            continue
        merged_scope_ids = _normalize_scope_ids(
            list(item.get("scope_ids", [])) + list(new_scope_ids)
        )
        if merged_scope_ids:
            normalized_entry["scope_ids"] = merged_scope_ids
        if item != normalized_entry:
            item.clear()
            item.update(normalized_entry)
            return True
        return False
    if new_scope_ids:
        normalized_entry["scope_ids"] = list(new_scope_ids)
    catalog.append(normalized_entry)
    return True


def detect_dataset_payload_path(dataset_dir: Path) -> str:
    payload_paths = detect_dataset_payload_paths(dataset_dir)
    return payload_paths[0] if payload_paths else ""


def detect_dataset_payload_paths(dataset_dir: Path) -> list[str]:
    data_dir = dataset_dir / "data"
    if data_dir.exists():
        files = sorted([p for p in data_dir.rglob("*") if p.is_file()])
        if files:
            return [str(path.relative_to(dataset_dir.parent)) for path in files]
    return []


def detect_dataset_download_script_path(dataset_dir: Path) -> str:
    download_script = dataset_dir / "download.py"
    if download_script.exists():
        return str(download_script.resolve())
    return ""


def dataset_directory_name(dataset_name: str) -> str:
    return str(dataset_name or "").strip() or "dataset"


def dataset_asset_id(scope_id: str, dataset_name: str) -> str:
    clean_scope = str(scope_id or "").strip() or "default"
    clean_name = str(dataset_name or "").strip() or "dataset"
    return f"{clean_scope}_{clean_name}"


def find_generated_dataset_dir(datasets_root: Path, dataset_name: str) -> Path | None:
    clean_name = dataset_directory_name(dataset_name)
    direct = datasets_root / clean_name
    if direct.is_dir():
        return direct
    return None


def _normalize_dataset_catalog_entry(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    normalized["name"] = str(normalized.get("name", "")).strip()
    normalized["description"] = str(normalized.get("description", "")).strip()
    normalized["downloaded_dataset_description"] = str(
        normalized.get("downloaded_dataset_description", "")
    ).strip()
    payload_files = normalized.get("payload_files")
    if not isinstance(payload_files, list):
        payload_files = []
    normalized["payload_files"] = [
        str(item).strip() for item in payload_files if str(item).strip()
    ]
    normalized["preview"] = _normalize_preview(normalized.get("preview", []))
    if "dataset_kind" not in normalized:
        normalized["dataset_kind"] = "real"
    normalized["has_ground_truth"] = bool(normalized.get("has_ground_truth", False))
    # Untagged entries (no scope_ids key) match all scopes in the cap fallback.
    if "scope_ids" in normalized:
        scope_ids = _normalize_scope_ids(normalized.get("scope_ids"))
        if scope_ids:
            normalized["scope_ids"] = scope_ids
        else:
            normalized.pop("scope_ids", None)
    return normalized


def _normalize_preview(preview: Any) -> list[dict[str, Any]]:
    if isinstance(preview, dict):
        files = preview.get("files", [])
        if isinstance(files, list):
            return [item for item in files if isinstance(item, dict)]
        return []
    if isinstance(preview, list):
        return [item for item in preview if isinstance(item, dict)]
    return []


def _normalize_reward_catalog_entry(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    name = str(normalized.get("name", "")).strip()
    normalized["name"] = name
    normalized["description"] = str(normalized.get("description", "")).strip()
    if "module" in normalized:
        normalized["module"] = str(normalized.get("module", "")).strip()
    if "class_name" in normalized:
        normalized["class_name"] = str(normalized.get("class_name", "")).strip()
    if "base_class" in normalized:
        normalized["base_class"] = str(normalized.get("base_class", "")).strip()
    if "default_params" in normalized:
        value = normalized.get("default_params")
        normalized["default_params"] = dict(value) if isinstance(value, dict) else {}
    for dropped in (
        "call_pattern",
        "category",
        "produced_ctx_keys",
        "init_ctx_args",
        "forward_ctx_args",
    ):
        normalized.pop(dropped, None)
    if "scope_ids" in normalized:
        scope_ids = _normalize_scope_ids(normalized.get("scope_ids"))
        if scope_ids:
            normalized["scope_ids"] = scope_ids
        else:
            normalized.pop("scope_ids", None)
    return normalized


def _normalize_scope_ids(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    seen: set[str] = set()
    for item in value:
        cleaned = str(item).strip()
        if cleaned:
            seen.add(cleaned)
    return sorted(seen)


def scope_match(entry: dict[str, Any], scope_ids: list[str] | None) -> bool:
    """Cap-accounting predicate: does *entry* count for the requested scopes?

    - `scope_ids` falsy (None / []): no scope filter — entry always counts.
    - Entry has no ``scope_ids`` key: legacy/untagged — counts for any scope
      (safe fallback so out-of-band catalog edits don't silently zero caps).
    - Otherwise: counts iff entry's scope_ids intersects requested scope_ids.
    """
    if not scope_ids:
        return True
    entry_scopes = entry.get("scope_ids")
    if entry_scopes is None:
        return True
    entry_set = {str(s).strip() for s in entry_scopes if str(s).strip()}
    if not entry_set:
        return True
    requested = {str(s).strip() for s in scope_ids if str(s).strip()}
    return bool(entry_set & requested)
