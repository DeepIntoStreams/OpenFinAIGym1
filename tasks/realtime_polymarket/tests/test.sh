#!/bin/bash
# Curated realtime-forecasting test.sh — same shape as offline tasks.
#
# The agent's train.py calls submit_predictions_async_and_persist; the
# curated verifier writes pending records to a SQLite ledger and the
# helper persists status_deferred=1.0 to reward.json. Final scoring
# happens via `python -m openfinai_harbor.resolve_deferred <trial_dir>`
# AFTER the prediction horizon elapses (typically 5 minutes).
set -e
mkdir -p /logs/verifier

python /data/run_evaluation_curated.py \
    --data-dir /data \
    --timeout 600 \
    --reward-output /logs/verifier/reward.json
