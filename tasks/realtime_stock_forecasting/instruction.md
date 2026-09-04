# Realtime Stock Forecasting (SPY, 5-minute horizon, Alpaca)

## Problem

Submit **price predictions** for `SPY` — the absolute future price you
expect `horizon_bars` ahead. Each prediction is **deferred**: the
verifier writes it to a SQLite ledger and returns immediately with
`status="deferred"`. Ground truth is resolved separately, after the
horizon elapses, by the resolver service (data source: Alpaca).

The horizon is `horizon_bars` bars of the task's `data_resolution`
interval — see [Resolution timing](#resolution-timing). The default config
(`horizon_bars = 5`, `data_resolution = "1m"`) targets the close **5
minutes** ahead.

Direction is **derived server-side** from
`sign(predicted_price - entry_price)`. The full metric panel after
resolution includes `price_mse`, `price_rmse`, `price_mae`,
`price_mape`, `price_r2`, `price_pearson`, `return_mse`, `return_rmse`,
`return_mae`, and `direction_accuracy`. Headline metric is configurable
via `task.toml`; default `price_mape`.

Legacy direction-only submissions still work but score on directional
metrics only (price metrics return NaN).

Requires `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` to be set on the
verifier host (the agent container does not need them — the verifier
fetches entry/exit prices itself).

## Resolution timing

The horizon is expressed in **bars**, not wall-clock minutes:

- `horizon_bars` (`task.toml [curated.default_config]`, default `5`) — how
  many bars ahead to predict.
- `data_resolution` (default `"1m"`) — the bar interval. Resolution is
  **coupled** to this: `1m` resolves on a minute close, `1h` on the hourly
  close, `1d` on the daily close.

**Entry** is the close of the bar containing your submission;
**exit** (ground truth) is the close of the bar `horizon_bars` later
(offline forecasting's `close.shift(-horizon_bars)`, in real time). With
the defaults, a submission at `13:00:30` resolves against the close of the
`13:05` minute bar.

`price_r2` / `price_pearson` need ≥ 2 ground-truth pairs **with variance**
in a session; single-prediction sessions — and coarse-horizon batches where
every prediction maps to the same close — report NaN for those two.

**Market hours caveat (equities only).** Bars only exist during the
trading session. If `horizon_bars × data_resolution` after your submission
lands outside market hours (overnight, weekend, holiday), the target bar
doesn't exist, the resolver marks that prediction `failed`, and it is
dropped from scoring. Submit comfortably before the close, or use a
`data_resolution`/`horizon_bars` that resolves within the session.

## Data

No static dataset shipped — realtime task. Submit a fixed strategy
or a deterministic heuristic. See the offline_stock_forecasting bundle
if you want historical price-action training data.

## Action format

Same shape as `realtime_crypto_forecasting`:

```python
predictions = [
    {
        "symbol": "SPY",
        "predicted_price": 502.5,    # absolute future price (recommended)
        # "direction":     "long",    # optional / derived from price
    },
    # ...
]
```

`entry_price` is intentionally NOT a field — the verifier captures it
server-side from the Alpaca feed at submission time, eliminating the
fake-price loophole that previously let an agent fabricate the entry.

## What you must build

```python
from harbor_submit import submit_predictions_async_and_persist
predictions = [{"symbol": "SPY", "predicted_price": 502.5}, ...]
submit_predictions_async_and_persist(predictions)
```

The helper writes a deferred reward.json + a `deferred_session.json`
sidecar. Run `python -m openfinai_harbor.resolve_deferred <trial_dir>`
after the horizon elapses (`horizon_bars × data_resolution`, ~5 min by
default) to populate the full metric panel and overwrite the reward.

## Hard constraints

- `predicted_price > 0` (or, on the legacy path, `direction` in
  `{"long", "short"}`). At least one signal must be supplied.
- `symbol` must match the configured set (default: `["SPY"]`).
- US market hours only — see the market-hours caveat under
  [Resolution timing](#resolution-timing); resolution fails if the target
  bar lands in a closed window.
- If you submit both `predicted_price` and `direction`, they must
  agree with the server-derived sign or the request is 422'd.

## Response format

Wrap your code in a fenced markdown block:

```python
# your code here
```

Output ONLY the markdown block — no surrounding prose.
