import csv
import json
from pathlib import Path


class ArtifactWriter:
    """file writer for JSON/CSV artifacts."""

    def __init__(self, artifact_dir: str) -> None:
        self._artifact_dir = Path(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, run_id: str, name: str, payload: dict) -> Path:
        path = self._artifact_dir / run_id / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def write_csv(self, run_id: str, name: str, rows: list[dict]) -> Path:
        path = self._artifact_dir / run_id / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        # Fieldnames = union of keys across ALL rows, preserving first-seen order.
        # Using only rows[0].keys() blew up when later rows had extra optional
        # keys (e.g. triage rows add `overlay_dir` only on routed/partial_match).
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path
