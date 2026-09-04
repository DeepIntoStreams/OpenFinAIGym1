# Realtime Polymarket Event Forecasting

## Problem

Submit calibrated **probability predictions** for active Polymarket
binary-event markets. Each market is a real-world YES/NO question
(politics, sports, crypto outcomes, …) with an absolute resolution
date; you forecast the probability of YES. After every market in the
session resolves, predictions are scored against the actual outcomes
using probabilistic-forecasting metrics.

Scoring is **deferred** — the verifier writes your predictions to a
SQLite ledger and returns immediately with `status="deferred"`.
Final scores arrive when an operator runs the resolver:

```bash
python -m openfinai_harbor.resolve_deferred <trial_dir>
```

The default discovery window selects markets resolving within the
next 1–6 hours, so the resolver typically completes within a few
hours of the trial. Markets that resolve to 0.5 (UMA dispute fallback) are
dropped from scoring and reported via the `n_dropped_ambiguous`
metadata field — they are unscoreable for calibration purposes.

## Universe (no static dataset)

This is a realtime task — there is no `dataset.h5` shipped into the
container. At task setup the verifier discovers the trial's universe
of active Polymarket markets matching the bundle's filters
(resolution window, volume, price extremes, categories). The
universe is **frozen for the trial** — re-fetching returns the same
set.

## Discovery flow (in-container)

```python
from harbor_submit import (
    fetch_active_markets,
    submit_event_predictions_async_and_persist,
)

# 1. Fetch the universe — list of market dicts the verifier discovered.
markets = fetch_active_markets()

# Each entry has:
#     symbol            (str — Polymarket conditionId, 0x-prefixed hash)
#     question          (str — the YES/NO question being asked)
#     description       (str — long-form context + resolution criteria)
#     resolution_at     (str — ISO-8601 UTC timestamp)
#     current_yes_price (float — current implied YES probability ∈ [0, 1])
#     best_bid, best_ask, spread  (orderbook NBBO; floats or None)
#     volume_24h, liquidity       (USD; floats or None)
#     tags, categories            (list[str])
#     outcomes                    (list[str], typically ["Yes", "No"])
#     slug                        (str — Polymarket URL slug)
```

## Action format

Build a list of prediction dicts. Submit **one batch per trial**:

```python
predictions = [
    {
        "symbol": markets[0]["symbol"],
        "predicted_yes_probability": 0.73,   # ∈ [0, 1]
    },
    {
        "symbol": markets[1]["symbol"],
        "predicted_yes_probability": 0.42,
    },
    # ... one per market you want to predict on
]

result = submit_event_predictions_async_and_persist(predictions)
# result["status"] == "deferred"
# result["deferred"]["session_id"] = …
```

Validation rules (verifier-side):

- `predicted_yes_probability` must be in `[0.0, 1.0]`.
- `symbol` must be one of the discovered universe (use
  `fetch_active_markets()` first).
- `entry_price` is **always** captured server-side — agent-supplied
  entry prices are silently ignored. The current YES price at
  submission time is recorded as the entry baseline.

## Metrics

After resolution the resolver computes (lower-is-better for all):

- **`brier_score`** — mean squared error between probability and
  outcome (∈ [0, 1]). **Headline metric**.
- **`log_loss`** — binary cross-entropy (-mean(y·log(p) + (1-y)·log(1-p))),
  with probabilities clipped to avoid log(0).
- **`expected_calibration_error`** — 10-bin ECE; measures whether
  e.g. all your "70% YES" predictions resolved YES ~70% of the time.

A uniform 0.5 prior scores `brier_score=0.25`, `log_loss=ln(2)≈0.693`.
Beat those numbers by extracting signal from the question text +
context.

## Time horizon

Default `task.toml` selects markets resolving 1–6 hours from now.
Adjust `[curated.default_config.discovery] resolution_window_hours_max`
to enlarge the window (e.g. weekly or monthly markets) — but be aware
the deferred resolver waits until the LATEST market in the session
resolves before scoring.

## Reference agent

`train.py` ships a uniform-prior baseline (predicts 0.5 for every
market) — replace it with an LLM agent that reads the question text
and reasons about the probability.

## Out of scope

This is a **forecasting** family — predictions only, no order
placement. Active polymarket trading (buying YES/NO shares for
profit) is a separate paradigm and may be added as a future
`realtime_polymarket_trading` bundle.
