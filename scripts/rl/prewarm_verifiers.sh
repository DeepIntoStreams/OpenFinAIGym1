#!/bin/bash
# Prewarm long-lived task verifiers for RL training.
# Usage: bash scripts/rl/prewarm_verifiers.sh <tasks-root>
# Re-running attaches to live verifiers instead of spawning duplicates.

set -euo pipefail

ROOT="${1:?usage: $0 <tasks-root> (e.g. tasks/)}"
if [[ ! -d "$ROOT" ]]; then
    echo "error: $ROOT is not a directory" >&2
    exit 1
fi

# Support both nested and flat task layouts.
mapfile -t TASK_TOMLS < <(find "$ROOT" -type f -name task.toml | sort)

if [[ ${#TASK_TOMLS[@]} -eq 0 ]]; then
    echo "warning: no task.toml files found under $ROOT — nothing to prewarm" >&2
    exit 0
fi

echo "=== Prewarming ${#TASK_TOMLS[@]} verifier(s) under $ROOT ==="

for toml in "${TASK_TOMLS[@]}"; do
    task_dir="$(dirname "$toml")"
    python - "$task_dir" <<'PY'
import os
import sys
from pathlib import Path
from openfinai_harbor.verifier.client import get_or_spawn

task_dir = Path(sys.argv[1]).resolve()
# Isolate registry files when concurrent jobs score the same task.
_root = os.environ.get("OPENFINAI_VERIFIER_RUN_OUTPUT_ROOT")
endpoint = get_or_spawn(
    task_dir,
    despawn_grace_min=1440.0,
    max_requests_per_minute=100000,
    run_output_root=Path(_root) if _root else None,
)
print(f"  {endpoint.task_id:40s} -> {endpoint.url}")
PY
done

echo "=== Prewarm complete ==="
