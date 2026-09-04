"""Reference baseline agent for realtime_polymarket.

Submits a uniform 0.5 prior for every discovered market — the
zero-information baseline. Real LLM agents replace the
``predict_probability`` body with reasoning over the market
question + description + resolution criteria.

Expected runtime environment (set by the curated bundle's
``run_evaluation_curated.py``):

    HARBOR_REWARD_OUTPUT  — path where reward.json should be written
    VERIFIER_URL          — http://host.docker.internal:<port>
    VERIFIER_TOKEN        — bearer token for the verifier RPC
    AGENT_ID              — uuid for this trial
"""

from __future__ import annotations

import os
import sys

# The bundle ships harbor_submit.py at /data/harbor_submit.py
# (mounted there by the curated Dockerfile). run_evaluation_curated.py adds
# /data to sys.path before invoking train.py.
sys.path.insert(0, "/data")
from harbor_submit import (  # noqa: E402
    fetch_active_markets,
    submit_event_predictions_async_and_persist,
)


def predict_probability(market: dict) -> float:
    """Return a probability ∈ [0, 1] that the market resolves YES.

    Baseline: uniform 0.5 prior. Override this in real agents — the
    market dict carries `question`, `description`, `current_yes_price`,
    `best_bid`, `best_ask`, `volume_24h`, `tags`, `categories`, etc.
    An LLM agent should read at least the question + description and
    reason about the probability.
    """
    return 0.5


def main() -> int:
    markets = fetch_active_markets()
    print(
        f"[train.py] discovered {len(markets)} markets",
        flush=True,
    )

    if not markets:
        print(
            "[train.py] empty universe — verifier discovery filters may "
            "have rejected all currently-active markets. Submitting "
            "empty batch is invalid; aborting with no submission.",
            file=sys.stderr,
        )
        return 1

    predictions = [
        {
            "symbol": m["symbol"],
            "predicted_yes_probability": predict_probability(m),
        }
        for m in markets
    ]

    reward_output = os.environ.get("HARBOR_REWARD_OUTPUT")
    result = submit_event_predictions_async_and_persist(
        predictions,
        reward_output_path=reward_output,
    )
    print(
        f"[train.py] submitted {len(predictions)} predictions "
        f"session_id={result.get('deferred', {}).get('session_id', 'n/a')[:8]} "
        f"status={result.get('status')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
