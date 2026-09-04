# Realtime Crypto Forecasting (BTCUSDT, 5-minute horizon)

## Problem

Submit **price predictions** for BTCUSDT — the absolute future price you
expect `horizon_bars` ahead. Each prediction is **deferred**: the
verifier writes it to a SQLite ledger and returns immediately with
`status="deferred"`. Ground truth (the actual exit price once the horizon
elapses) is resolved by the resolver service, which then computes the full
metric panel (price-error + directional metrics).

The horizon is `horizon_bars` bars of the task's `data_resolution`
interval — see [Resolution timing](#resolution-timing). The default config
(`horizon_bars = 5`, `data_resolution = "1m"`) targets the close **5
minutes** ahead.

Direction is **derived server-side** from
`sign(predicted_price - entry_price)`, so a single price prediction
produces both price-regression scores (mse / rmse / mae / mape / r2 /
pearson) and `direction_accuracy`. The trial **headline** (the `reward`
value in `reward.json`) is configurable via
`task.toml [curated.default_config] headline_metric`; the default is
`price_mape` (mean absolute percentage error).

Legacy direction-only submissions (no `predicted_price`, just `direction`)
still work — they score on `direction_accuracy` only, with all price
metrics returning `NaN`.

## Resolution timing

The horizon is expressed in **bars**, not wall-clock minutes:

- `horizon_bars` (`task.toml [curated.default_config]`, default `5`) — how
  many bars ahead to predict.
- `data_resolution` (default `"1m"`) — the bar interval. Resolution is
  **coupled** to this: a `1m` data resolution resolves on a minute close,
  `1h` on the hourly close, `1d` on the daily close.

**Entry** is the close of the bar containing your submission time.
**Exit** (ground truth) is the close of the bar `horizon_bars` later —
exactly like offline forecasting's `close.shift(-horizon_bars)`, but in
real time. With the default `horizon_bars = 5`, `data_resolution = "1m"`,
a submission at `12:00:30` resolves against the close of the `12:05` minute
bar (≈ the price at `12:06:00`).

Because resolution snaps to a bar boundary, the realised horizon can differ
from the nominal one by up to the submission's offset within its bar (< one
bar). To resolve on a coarser canonical close, the task admin sets
`data_resolution = "1h"` or `"1d"` (which also makes the agent's observed
bars hourly/daily — observation and resolution share the same interval).

## Data (no static dataset)

This is a **realtime** task — there is no `dataset.h5` shipped into the
container. The agent has no offline data to train on. You can either:

1. Submit a fixed strategy (e.g. always `predicted_price = entry * 1.001`)
   as a baseline.
2. Use a deterministic heuristic with hardcoded parameters. (No external
   network access from inside the container.)

Note: harbor's Trial / Docker path does not have network access from the
agent container — only the host-side verifier does.

## Action format

Build a list of prediction dicts. Submit **one batch per trial**:

```python
predictions = [
    {
        "symbol": "BTCUSDT",
        "predicted_price": 68500.0,   # absolute future price (recommended)
        # "direction":     "long",     # optional / legacy; auto-derived from price
    },
    # ... more predictions ...
]
```

**Use `predicted_price` for the full metric panel.** If you submit
`direction` only (legacy path), price metrics report NaN.

**`entry_price` is intentionally NOT a field.** The verifier captures
the entry price server-side from the live Binance feed at the moment
your submission is received, then captures the exit price the same way
after the horizon elapses. Agent-supplied entry prices used to be honored
but were dropped because they enabled a fake-price exploit. The
forward-looking `predicted_price` does NOT enable the same exploit — an
agent submitting an absurd value just gets a bad score, not free PnL.

## Validation

The handler enforces, in order:
1. At least one of `predicted_price` or `direction` must be supplied
   (pydantic 422 if both are missing).
2. `predicted_price > 0` (pydantic 422 on ≤ 0).
3. If both `predicted_price` and `direction` are supplied, they must
   agree with the server-derived sign — mismatched submissions are
   rejected with a 422 carrying the conflicting values.
4. `symbol` must be in the configured target subset (default
   `["BTCUSDT"]`); other symbols are 422'd.

## What you must build

Write `/workspace/train.py` that:

1. Builds a list of prediction dicts (5–20 is reasonable for a smoke
   trial). Use a deterministic strategy — single-shot LLMs cannot
   actually forecast crypto prices, so the goal is to exercise the
   deferred-ledger contract and the price-prediction wire format.
2. Submits via the in-container helper:
   ```python
   from harbor_submit import submit_predictions_async_and_persist
   result = submit_predictions_async_and_persist(predictions)
   print("session:", result["deferred"]["session_id"])
   ```
3. The helper writes a `reward.json` with `status_deferred=1.0` and
   `reward=0.0` so harbor's verifier step parses cleanly. The actual
   `reward` value (and the full metric panel) gets overwritten after
   `python -m openfinai_harbor.resolve_deferred <trial_dir>` runs.

## Hard constraints

- **`predicted_price` must be > 0** (positive absolute price).
- **`symbol` must be one of the configured symbols** (default:
  `BTCUSDT`). Other symbols are rejected with 422.
- **No torch / GPU**. Keep total compute time well under 60s.

## Evaluation flow

1. Trial runs → agent submits predictions → verifier writes ledger →
   reward.json has `status_deferred=1.0`, `reward=0.0`.
2. **Wait for the horizon to elapse** (`horizon_bars × data_resolution`
   per the task config, default 5 × 1m = 5 minutes).
3. Run `python -m openfinai_harbor.resolve_deferred <trial_dir>`.
   This resolves the ledger via the data provider and runs the
   per-session aggregator that produces all metrics in one pass:
   `price_mse`, `price_rmse`, `price_mae`, `price_mape`, `price_r2`,
   `price_pearson`, `return_mse`, `return_rmse`, `return_mae`,
   `direction_accuracy`. The headline (configured via `task.toml`)
   becomes the `reward` value.
4. Read the updated `reward.json` for the final headline + per-metric
   scores.

Note: `price_r2` and `price_pearson` need ≥ 2 valid (price, ground-truth)
pairs **with variance** in the same session; single-prediction sessions
report NaN for those two. A coarse horizon (e.g. `1d`) where every
prediction in one batch maps to the same daily close also collapses the
ground-truth variance — those two metrics degenerate to NaN there too.

## Response format

Wrap your code in a fenced markdown block:

```python
# your code here
```

Output ONLY the markdown block — no surrounding prose.
