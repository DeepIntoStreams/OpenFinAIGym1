#!/bin/bash
# Curated trading test.sh — delegates to run_evaluation_curated.py.
#
# Same shape as the forecasting bundle: the runner runs train.py inside
# the container with HARBOR_REWARD_OUTPUT plumbed through, and the agent
# is expected to call harbor_submit.submit_actions_and_persist
# (NOT submit_and_persist — trading uses an action stream,
# not a prediction array). The curated verifier replays the action
# stream on the host and writes reward.json.
set -e
mkdir -p /logs/verifier

python /data/run_evaluation_curated.py \
    --data-dir /data \
    --timeout 600 \
    --reward-output /logs/verifier/reward.json
