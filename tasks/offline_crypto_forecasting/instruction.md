# Offline Crypto Forecasting (BTCUSDT / ETHUSDT / SOLUSDT, hourly)

## Problem

Predict the **absolute close price** `forecast_horizon_bars` ahead for
each input symbol given that bar's 10-dimensional OHLCV-derived feature
vector. Hourly bars from Binance, sliced chronologically into a
train/test window.

Default horizon is 1 bar (next-bar prediction); the task admin can
override via `task.toml [curated.default_config] forecast_horizon_bars`.

The metric panel emitted after scoring is:

| Per symbol | Macro (mean over `target_symbols`) |
|------------|-------------------------------------|
| `mse_<sym>`, `rmse_<sym>`, `mae_<sym>`, `mape_<sym>`, `r2_<sym>`, `pearson_<sym>`, `directional_accuracy_<sym>` | `mape`, `r2`, `pearson`, `directional_accuracy` |

The macro keys cover only the **scale-free** metrics — raw MSE/MAE/RMSE
on a multi-symbol task can't be macro-averaged because BTC ≈ 60k and
SOL ≈ 100 live on different scales. Per-symbol versions are still
emitted for every input symbol so you can read absolute errors per
asset.

The trial **headline** (the `reward` value in `reward.json`) is
configurable via `task.toml`; default `mape`.

`directional_accuracy` is *derived* — for each row,
`sign(predicted_price - reference_price)` is compared against the
ground-truth `sign(future_price - reference_price)`. The reference price
is the current bar's close, exposed via `task.get_reference_test()`.

## Data access (same contract as auto-pipeline tasks)

The bundle is mounted at `/data/`. Import the bundled task class and use
its accessors — **do not** parse `dataset.h5` by hand, and do not fetch
live data:

```python
from task import OfflineCryptoForecasting

task = OfflineCryptoForecasting()
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

The feature column order (identical for every symbol) is
`task.FEATURE_NAMES`:

```
return_1h, log_return_1h,
rolling_mean_5h, rolling_std_5h,
rolling_mean_20h, rolling_std_20h,
volume_change, high_low_range, close_open_range, momentum_20h
```

`list(X_test)` enumerates the symbols. The default config has 3 symbols
(`BTCUSDT`, `ETHUSDT`, `SOLUSDT`), of which 2 are scored (`BTCUSDT`,
`ETHUSDT`); per-symbol metrics are emitted for all of them.

> **Underlying file (reference only).** The accessors above read a
> symbol-first HDF5 at `/data/dataset.h5`: groups `/<sym>/X_train`,
> `/<sym>/y_train`, `/<sym>/X_test`, `/<sym>/reference_train`,
> `/<sym>/reference_test`, plus root attrs `layout="per_symbol"`,
> `feature_names`, `forecast_horizon_bars`, `headline_metric`. The
> held-out `y_test` is **not** in this file. Prefer the accessors.

## What you must build

Write `/workspace/train.py` that:

1. Loads the task via the accessors above.
2. Fits a small, deterministic regressor per symbol on
   `(X_train[s], y_train[s])`. Prefer:
   - `sklearn.linear_model.Ridge` / `LinearRegression`
   - `sklearn.ensemble.GradientBoostingRegressor` (small)
   - Numpy baselines like `reference * (1 + drift)` (the reference
     array makes this trivial).
   No torch / tensorflow / GPU. Keep training under 60 seconds.
3. Predicts on `X_test[s]` (shape `(n_test,)`). Output **absolute
   prices**, not log-returns or relative changes.
4. Submits via the in-container helper:
   ```python
   from harbor_submit import submit_and_persist
   predictions = {s: model[s].predict(X_test[s]) for s in X_test}
   submit_and_persist(predictions)
   ```
   The helper handles networking, auth, and writing `reward.json`.

## Submission format

Submissions are **always dict-keyed-by-symbol** —
`{"BTCUSDT": np.ndarray((n_test,)), "ETHUSDT": ...}`. The helper
encodes via `np.savez` and posts to `/submit/predictions`. Sending a
flat ndarray is a contract violation because `_score_dict` dispatches
on `isinstance(predictions, dict)`.

## Evaluation

The verifier scores via the gt-at-init torch `Loss` panel (`MSELoss`,
`RMSELoss`, `MAELoss`, `MAPELoss`, `R2Score`, `PearsonCorrelation`)
plus `DirectionalAccuracy` on signed price-vs-reference diffs, against
the held-out target. The configured headline metric (default `mape`)
becomes the `reward` value.

## Hard constraints

- Predictions must be **absolute prices**, not relative changes.
- Predictions must be finite. NaN / inf in your prediction array poison
  the metric panel — the gt-at-init Loss classes propagate NaN through
  `forward()`.
- Length: `predictions[<sym>].shape[0]` must equal
  `task.get_features()[<sym>].shape[0]`. Extra trailing values are
  ignored; missing values produce zero per-row contributions and tank
  the metric.
- Do **not** call `task.get_ground_truth()` (it raises) and do **not**
  write `predictions.npy` / `reward.json` yourself — the submit helper
  does it.

## Response format

Wrap your code in a fenced markdown block:

```python
# your code here
```

Output ONLY the markdown block — no surrounding prose.
