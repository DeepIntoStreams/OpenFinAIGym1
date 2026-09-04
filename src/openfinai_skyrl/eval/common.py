"""Helpers shared by baseline + trained execution evals."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openfinai_skyrl.data.splitting import DEFAULT_TEST_TASKS


AGENT_BY_MODE: dict[str, str] = {
    "single": "openfinai_harbor.agents:SingleShotLLMAgent",
    "multi": "openfinai_harbor.agents:ReActFinAgent",
}


def read_reward(trial_dir: Path) -> float | None:
    """Read the per-trial reward written by Harbor's verifier.

    Probes (in order) ``verifier/reward.json`` → ``verifier/reward.txt`` →
    ``result.json``'s ``verifier_result`` dict. Returns ``None`` if no
    parseable reward is found.
    """
    for candidate in (trial_dir / "verifier" / "reward.json", trial_dir / "verifier" / "reward.txt"):
        if not candidate.exists():
            continue
        try:
            if candidate.suffix == ".json":
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "reward" in data:
                    return float(data["reward"])
            else:
                return float(candidate.read_text().strip())
        except (ValueError, json.JSONDecodeError, OSError):
            continue
    result_path = trial_dir / "result.json"
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            vr = (payload.get("verifier_result") or {}) if isinstance(payload, dict) else {}
            rewards = vr.get("rewards") or {}
            if "reward" in rewards:
                return float(rewards["reward"])
            if "reward" in vr:
                return float(vr["reward"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return None


def newest_trial_dir(trials_dir: Path) -> Path | None:
    """Return the most recently-modified subdirectory of ``trials_dir`` (or None)."""
    if not trials_dir.exists():
        return None
    candidates = [p for p in trials_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_test_tasks(cli_tasks: list[str] | None, tasks_root: Path) -> list[Path]:
    """Resolve task names to existing directories under ``tasks_root``.

    Falls back to :data:`DEFAULT_TEST_TASKS` when ``cli_tasks`` is empty.
    Raises ``SystemExit`` if any requested name has no matching directory.
    """
    names = list(cli_tasks) if cli_tasks else list(DEFAULT_TEST_TASKS)
    resolved: list[Path] = []
    for name in names:
        candidate = tasks_root / name
        if not candidate.exists():
            raise SystemExit(f"Test task missing under {tasks_root}: {name}")
        resolved.append(candidate)
    return resolved


def system_prompt_from_dataset_manifest(dataset_dir: str | None) -> str | None:
    """Look up ``system_prompt_path`` from ``dataset_manifest.json``.

    Lets eval-time match SFT prep byte-for-byte without the operator
    repeating the prompt path. Returns ``None`` on any failure (missing
    dir / manifest / key) so callers can fall through to defaults.
    """
    if not dataset_dir:
        return None
    ds_path = Path(dataset_dir)
    for candidate in (
        ds_path / "dataset_manifest.json",
        ds_path.parent / "dataset_manifest.json",
    ):
        if candidate.exists():
            try:
                manifest = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            recorded = manifest.get("system_prompt_path") if isinstance(manifest, dict) else None
            # ``prepare.py`` writes the sentinel ``"<trajectory-recorded>"`` when no
            # ``--system-prompt-path`` was passed (i.e. the system prompt was
            # taken from each trial's trajectory verbatim). Don't surface
            # sentinels to the agent — they would be interpreted as file paths.
            if isinstance(recorded, str) and recorded and not (
                recorded.startswith("<") and recorded.endswith(">")
            ):
                return recorded
    return None


def resolve_agent_import_path(mode: str | None, manifest: dict[str, Any]) -> str:
    """Pick the agent class import path for an eval run.

    Prefers an explicit ``mode``; falls back to the value recorded in
    the train manifest. Raises ``SystemExit`` if neither names a known mode.
    """
    if mode is None:
        manifest_mode = manifest.get("mode")
        if manifest_mode not in AGENT_BY_MODE:
            raise SystemExit(
                "--mode not provided and run manifest does not name a single|multi mode. "
                f"Manifest mode={manifest_mode!r}."
            )
        return AGENT_BY_MODE[manifest_mode]
    return AGENT_BY_MODE[mode]


__all__ = [
    "AGENT_BY_MODE",
    "newest_trial_dir",
    "read_reward",
    "resolve_agent_import_path",
    "resolve_test_tasks",
    "system_prompt_from_dataset_manifest",
]
