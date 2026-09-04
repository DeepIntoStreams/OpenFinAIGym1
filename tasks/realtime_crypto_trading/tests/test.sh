#!/bin/bash
set -e
mkdir -p /logs/verifier
python /data/run_evaluation_curated.py \
    --data-dir /data \
    --timeout 600 \
    --reward-output /logs/verifier/reward.json
