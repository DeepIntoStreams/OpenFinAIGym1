# Offline Stock Forecasting (SPY / QQQ / IWM, hourly, Alpaca)

## Problem

Predict the **absolute close price** `forecast_horizon_bars` ahead for
each input ticker given that bar's 10-dimensional OHLCV-derived feature
vector. Hourly bars from Alpaca (free IEX tier), sliced
chronologically into a train/test window.

Default horizon is 1 bar (next-bar prediction); admin-configurable via
`task.toml`.

Metric panel + headline behaviour are identical to the
`offline_crypto_forecasting` bundle — see that instruction file for the
full panel breakdown. Default headline is `mape`. Macro keys (`mape`,
`r2`, `pearson`, `directional_accuracy`) are the unweighted mean across
`target_symbols`. Per-symbol absolute MSE/MAE/RMSE are emitted but not
macro-averaged because cross-ticker price scales differ (SPY ≈ 500 vs
IWM ≈ 200).

## Data access (same contract as auto-pipeline tasks)

The bundle is mounted at `/data/`. Import the bundled task class and use
its accessors — **do not** parse `dataset.h5` by hand, and do not fetch
live data:

```python
from task import OfflineStockForecasting

task = OfflineStockForecasting()
task.load_data()                       # reads /data/dataset.h5 via load.py

X_train = task.get_train_features()       # {sym: (n_train, 10)}
y_train = task.get_train_ground_truth()   # {sym: (n_train,)}  absolute future close prices
X_test  = task.get_features()             # {sym: (n_test, 10)}  PREDICT ON THESE
ref_test = task.get_reference_test()      # {sym: (n_test,)}  current-bar close (for relative baselines)
```

`task.get_ground_truth()` **raises `PermissionError`** by design — the
held-out test target stays verifier-side (in
`environment/eval-data/test_ground_truth.h5`, which your container never
mounts). There is no in-container self-evaluation; the verifier scores
your submission. Fit on the train split.

The feature column order (`task.FEATURE_NAMES`, identical for every
ticker) is:

```
return_1h, log_return_1h,
rolling_mean_5h, rolling_std_5h,
rolling_mean_20h, rolling_std_20h,
volume_change, high_low_range, close_open_range, momentum_20h
```

`list(X_test)` enumerates the tickers. The default config has 3 tickers
(`SPY`, `QQQ`, `IWM`), of which 2 feed the macro aggregate (`SPY`,
`QQQ`); per-symbol metrics are emitted for all of them.

> **Underlying file (reference only).** The accessors read a symbol-first
> HDF5 at `/data/dataset.h5`: groups `/<sym>/X_train`, `/<sym>/y_train`,
> `/<sym>/X_test`, `/<sym>/reference_train`, `/<sym>/reference_test`,
> plus root attrs `layout="per_symbol"`, `feature_names`,
> `forecast_horizon_bars`, `headline_metric`. The held-out `y_test` is
> **not** in this file. Prefer the accessors.

## What you must build

Same shape as the offline_crypto_forecasting bundle — substitute the
ticker names. Write `/workspace/train.py` that:

1. Loads the task via the accessors above.
2. Fits a deterministic regressor per symbol on `(X_train, y_train)`.
   No torch / tensorflow / GPU; keep training under 60 seconds.
3. Predicts **absolute prices** on `X_test` (one prediction per test
   row, dict keyed by symbol).
4. Submits via:
   ```python
   from harbor_submit import submit_and_persist
   predictions = {s: model[s].predict(X_test[s]) for s in X_test}
   submit_and_persist(predictions)
   ```

## Submission format

Always dict-keyed-by-symbol —
`{"SPY": np.ndarray((n_test,)), "QQQ": ..., "IWM": ...}`. Sending a
flat ndarray is a contract violation.

## Evaluation

The verifier scores via the gt-at-init `Loss` panel (`MSELoss`,
`RMSELoss`, `MAELoss`, `MAPELoss`, `R2Score`, `PearsonCorrelation`)
plus `DirectionalAccuracy` on signed price-vs-reference diffs, against
the held-out target. The headline metric (default `mape`) becomes
`reward`.

## Hard constraints

- Predictions must be **absolute prices**, not log-returns / relative
  changes.
- Submit via the helper, not by writing `predictions.npy` / `reward.json`.
- Predictions must be finite. NaN / inf poison the metric panel.
- Length: `predictions[<sym>].shape[0]` must match
  `task.get_features()[<sym>].shape[0]`.
- Do **not** call `task.get_ground_truth()` (it raises).

## Response format

Wrap your code in a fenced markdown block:

```python
# your code here
```

Output ONLY the markdown block — no surrounding prose.
