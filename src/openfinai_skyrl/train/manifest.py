"""Run-metadata helpers used to populate the SFT manifest."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def resolve_git_sha() -> str | None:
    """Return repo HEAD short SHA or None; best-effort, never raises."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[4]),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    sha = result.stdout.strip()
    return sha or None


def hash_config(raw_config: dict[str, Any]) -> str:
    """SHA-256 of the raw config dict (sorted-key JSON canonicalisation)."""
    blob = json.dumps(raw_config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def snapshot_dataset_system_prompt(dataset_name: str, dataset_split: str) -> dict[str, Any]:
    """Return ``{chars, sha256, preview}`` for the split's first system message, or ``{}``."""
    try:
        from openfinai_skyrl.train.tokenization import load_messages_dataset

        ds = load_messages_dataset(dataset_name, dataset_split)
        if len(ds) == 0:
            return {}
        msgs = ds[0].get("messages") or []
        first = next((m for m in msgs if m.get("role") == "system"), None)
        if not first:
            return {}
        content = str(first.get("content", ""))
        if not content:
            return {}
        return {
            "chars": len(content),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "preview": content[:200],
        }
    except Exception:
        return {}


__all__ = ["hash_config", "resolve_git_sha", "snapshot_dataset_system_prompt"]
