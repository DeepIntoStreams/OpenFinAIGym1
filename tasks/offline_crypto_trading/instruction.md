# Offline Crypto Trading (BTCUSDT, hourly replay)

## Problem

Sequential trading on **historical** Binance hourly bars. You submit orders
(buy / sell / hold, with optional limit / stop types) to manage a
buying-power-capped portfolio. Shorting is allowed.

Score is **PnL** (mark-to-market profit-and-loss across the episode).
Diagnostic metrics: Sharpe ratio, max drawdown, win rate, per-bar return
statistics.

## How the episode runs (interactive — no data file)

The verifier holds the market data and replays it **one bar at a time**. You
do **not** receive the price series up front, and there is no `dataset.h5`.
Instead you drive an interactive session:

1. `reset` → the verifier returns the **first observation** (the current bar
   plus recent history up to *now*).
2. You decide an action for the current bar and `step` it.
3. The verifier advances one bar and returns the **next observation** plus
   the realized reward — and on the final bar, the score.

You only ever see bars up to the current step, so there is no way to look
ahead. Scoring happens server-side.

### Observation shape

Each observation is a dict (the same shape the realtime trading tasks use;
`order_book` is always `None` for historical replay):

```python
{
  "step": int,                 # current bar index within the episode
  "steps_remaining": int,
  "symbols": {
    "BTCUSDT": {
      "symbol": "BTCUSDT", "price": float,
      "open": float, "high": float, "low": float, "close": float,
      "volume": float, "timestamp": str,
      "return_1h": float, "returns_5h": [float, ...], "returns_20h": [float, ...],
      "recent_bars": [ {open, high, low, close, volume, ...}, ... ],
      "order_book": None,
    }
  },
  "portfolio": {
    "cash": float, "reserved_cash": float,
    "positions": {sym: float},          # only symbols you hold (use .get(sym, 0.0))
    "pnl": {sym: float, "__total": float},
    "value": float, "pending_orders": [...],
  },
}
```

## Action format

One action **per bar**, a transactional order intent:

```python
{
  "action": "buy" | "sell" | "hold" | "cancel",
  "symbol": "BTCUSDT",          # required for buy/sell
  "quantity": float,            # > 0, fractional allowed
  "order_type": "market" | "limit" | "stop" | "stop_limit",   # default "market"
  "limit_price": float,         # required for limit / stop_limit
  "stop_price": float,          # required for stop / stop_limit
  "tif": "ioc" | "gtc",         # default "gtc"; ignored for market/hold
  "order_id": str,              # required only for action="cancel"
}
```

Submit several at once with `{"orders": [<intent>, ...]}`. `{"action": "hold"}`
does nothing this bar. **Orders are capped by buying power (strict 1×)** — size
quantity to your cash, e.g. `qty = 0.95 * cash / price`. A `sell` beyond your
holdings opens a short. Limit/stop orders queue across bars (GTC) or expire the
same bar (IOC); fills are pessimistic at the limit/stop price.

## What you must build

Write `/workspace/train.py` that defines a per-bar **policy** and runs the
session via the bundled helper:

```python
from harbor_submit import run_trading_session

def policy(obs):
    bar = obs["symbols"]["BTCUSDT"]
    cash = obs["portfolio"]["cash"]
    pos = obs["portfolio"]["positions"].get("BTCUSDT", 0.0)
    r5 = bar.get("returns_5h") or []
    if sum(r5) > 0 and pos <= 0:           # momentum up & not long → go long
        qty = 0.95 * cash / bar["close"]
        return {"action": "buy", "symbol": "BTCUSDT", "quantity": qty}
    if sum(r5) < 0 and pos > 0:            # momentum down & long → flatten
        return {"action": "sell", "symbol": "BTCUSDT", "quantity": pos}
    return {"action": "hold"}

run_trading_session(policy)   # loops reset -> step* and writes reward.json
```

`run_trading_session` handles the reset/step loop, networking, auth, and
writing `reward.json`. If you prefer, drive `session_reset()` /
`session_step(session_id, action)` directly instead.

## Hard constraints

- **Decide each action from the observation you are given** — you cannot
  fetch future bars; the verifier withholds them.
- **Size orders to your buying power** — orders that exceed it are rejected
  (see `info["rejections"]`); they do not raise.
- **No torch / GPU / large libraries.** NumPy + pandas + sklearn are
  available; keep total compute well within the episode budget.
- **Do not write `reward.json` yourself** — `run_trading_session` writes it.

## Evaluation

Reward keys (computed server-side):

- `pnl` — total PnL across the episode. **This is the headline metric**
  (= `reward` in `reward.json`).
- `sharpe_ratio` — annualised Sharpe of the per-bar return stream.
- `max_drawdown` — largest peak-to-trough drawdown.
- `win_rate` — fraction of bars with positive return.
- `total_return`, `mean_return`, `std_return`, `num_trades`, `total_pnl` —
  scalar stats from the trade history.
- `steps_executed` — number of bars stepped (should equal the episode length).

## Response format

Wrap your code in a fenced markdown block:

```python
# your code here
```

Output ONLY the markdown block — no surrounding prose.
