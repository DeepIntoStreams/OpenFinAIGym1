#!/bin/bash
# Minimal test script — calls the shared evaluation runner.
# The assembled evaluator is discovered automatically from the task name.
# Optionally pass --weights to override the default equal weighting.
set -e
mkdir -p /logs/verifier

python /openfinai/eval/run_evaluation.py \
    --task your_task_name \
    --data-dir /data \
    --reward-output /logs/verifier/reward.json
