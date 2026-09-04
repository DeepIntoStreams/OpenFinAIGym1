#!/bin/bash
# Curated forecasting test.sh — delegates to run_evaluation_curated.py.
#
# The runner runs the LLM-generated /workspace/train.py inside the
# container with HARBOR_REWARD_OUTPUT plumbed through to the canonical
# /logs/verifier/reward.json mount. The agent's train.py is expected
# to call harbor_submit.submit_and_persist(...) so
# the reward lands at HARBOR_REWARD_OUTPUT.
#
# This script does NOT load any local evaluator — the curated verifier
# RPC service holds the held-out test target on the host and scores
# remotely. R3 isolation is enforced by the network boundary.
set -e
mkdir -p /logs/verifier

python /data/run_evaluation_curated.py \
    --data-dir /data \
    --timeout 600 \
    --reward-output /logs/verifier/reward.json
