# Offline Stock Trading (SPY, hourly replay)

## Problem

Sequential trading on **historical** Alpaca hourly bars for `SPY`. Same
protocol as `offline_crypto_trading`: you submit transactional orders
(buy / sell / hold, with optional limit / stop types) to manage a
buying-power-capped portfolio; shorting is allowed; scored server-side.
Headline metric is `pnl`.

## How the episode runs (interactive — no data file)

The verifier holds the market data and replays it **one bar at a time**.
There is no `dataset.h5`; you do **not** receive the price series up front.
You drive an interactive session: `reset` returns the first observation,
then each `step(action)` advances one bar and returns the next observation
(plus the realized reward, and the score on the final bar). You only ever
see bars up to the current step — no look-ahead is possible.

### Observation shape

Same shape as the realtime trading tasks (`order_book` is `None` for
historical replay):

```python
{
  "step": int, "steps_remaining": int,
  "symbols": {"SPY": {"symbol","price","open","high","low","close","volume",
                      "timestamp","return_1h","returns_5h":[...],"returns_20h":[...],
                      "recent_bars":[...], "order_book": None}},
  "portfolio": {"cash","reserved_cash","positions":{sym: float},
                "pnl":{sym: float,"__total": float},"value","pending_orders":[...]},
}
```

`positions` only contains symbols you currently hold — use
`positions.get(sym, 0.0)`.

## Action format

One action **per bar**, a transactional order intent:

```python
{
  "action": "buy" | "sell" | "hold" | "cancel",
  "symbol": "SPY",              # required for buy/sell
  "quantity": float,            # > 0, fractional allowed
  "order_type": "market" | "limit" | "stop" | "stop_limit",   # default "market"
  "limit_price": float, "stop_price": float,   # for limit / stop / stop_limit
  "tif": "ioc" | "gtc",         # default "gtc"; ignored for market/hold
  "order_id": str,              # required only for action="cancel"
}
```

Submit several at once with `{"orders": [<intent>, ...]}`; `{"action": "hold"}`
is a no-op. **Orders are capped by buying power (strict 1×)** — size quantity to
your cash (`qty = 0.95 * cash / price`). A `sell` beyond your holdings opens a
short. GTC limit/stop orders carry across bars and auto-expire at episode end.

## What you must build

Write `/workspace/train.py` that defines a per-bar **policy** and runs the
session via the bundled helper:

```python
from harbor_submit import run_trading_session

def policy(obs):
    bar = obs["symbols"]["SPY"]
    cash = obs["portfolio"]["cash"]
    pos = obs["portfolio"]["positions"].get("SPY", 0.0)
    r5 = bar.get("returns_5h") or []
    if sum(r5) > 0 and pos <= 0:
        return {"action": "buy", "symbol": "SPY", "quantity": 0.95 * cash / bar["close"]}
    if sum(r5) < 0 and pos > 0:
        return {"action": "sell", "symbol": "SPY", "quantity": pos}
    return {"action": "hold"}

run_trading_session(policy)   # loops reset -> step* and writes reward.json
```

`run_trading_session` handles the reset/step loop, networking, auth, and
writing `reward.json`. You can also drive `session_reset()` /
`session_step(session_id, action)` directly.

## Hard constraints

- **Decide each action from the observation you are given** — future bars
  are withheld by the verifier.
- **Size orders to your buying power** — over-budget orders are rejected
  (`info["rejections"]`), not raised.
- **No torch / GPU / large libraries**; NumPy + pandas + sklearn available.
- **Do not write `reward.json` yourself** — `run_trading_session` writes it.

## Evaluation

Headline metric: `pnl` (= `reward`). Diagnostics: `sharpe_ratio`,
`max_drawdown`, `win_rate`, `total_return`, `mean_return`, `std_return`,
`num_trades`, `total_pnl`, `steps_executed`.

## Response format

Wrap your code in a fenced markdown block:

```python
# your code here
```

Output ONLY the markdown block — no surrounding prose.
