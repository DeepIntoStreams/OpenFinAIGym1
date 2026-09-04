# Realtime Crypto Trading (BTCUSDT/ETHUSDT/SOLUSDT, internal-paper, Binance feed)

## Problem

Run a real-time paper-trading episode against the live Binance public
market-data feed. Each step you receive a fresh market observation and
return one or more orders; the bundled in-process `SimulatedExecutor`
(execution mode `internal_paper`) fills them against the live price and
produces an immediate mark-to-market PnL reward. After the episode ends,
metrics are aggregated by the curated reward bank and written to
`reward.json`.

- **Data feed:** Binance public REST + WebSocket (no API key needed).
- **Execution:** in-process `SimulatedExecutor`. Fills are simulated
  locally against the observed price with slippage `0.1%`
  (`slippage_pct = 0.001`) and `0%` transaction cost. Starting cash is
  **100,000** (quote currency, e.g. USDT).
- **Market:** crypto trades 24/7 — no market-hours restriction.
- **Reward:** per-step change in total mark-to-market PnL.

## What you write

A single Python file at `/workspace/train.py` exposing a top-level
callable:

```python
def step(observation: dict) -> dict:
    """Decide one action (or a batch of orders) for this step.

    Parameters
    ----------
    observation : dict
        See "Observation shape" below.

    Returns
    -------
    dict
        A single order dict, OR a batch {"orders": [<order>, ...]}.
        See "Action format" below.
    """
    ...
```

Anything else in `train.py` (imports, helpers, module-level state) is
yours. The bundled runner (`/data/run_evaluation_curated.py`) imports
`step` by name and drives the gym loop — you do **not** loop yourself,
do **not** call `task.reset()`, do **not** write `reward.json`. The
runner owns all of that so the action-vs-scoring boundary stays clean.

## Observation shape

The runner serialises everything to plain Python dicts before calling
your `step()` — no dataclasses, no datetime objects, just dict-style
access. Bars are plain dicts, so `bar["close"]` and `bar.get("price")`
both work.

```python
{
    "step": 3,                  # steps taken so far
    "steps_remaining": 7,       # -1 if max_steps is unbounded
    "symbols": {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "price": 65020.50,                     # latest price (float | None)
            "volume": 12.34,                        # in-progress bar's running
                                                    #   cumulative volume (float | None);
                                                    #   resets at each 1m boundary
            "timestamp": "2026-05-09T05:21:00+00:00",  # ISO-8601 of latest bar (str | None)
            "recent_bars": [                        # last 60 1m bars, chronological
                {
                    "symbol": "BTCUSDT",
                    "timestamp": "2026-05-09T05:20:00+00:00",
                    "price": 65010.0,
                    "open": 65005.0, "high": 65025.0,
                    "low": 65000.0, "close": 65010.0,
                    "volume": 12.34,
                    "extra": {},
                },
                # ...
            ],
            "order_book": {                         # OrderBookSnapshot | None
                "symbol": "BTCUSDT",
                "timestamp": "2026-05-09T05:21:00+00:00",
                "best_bid": 65020.0, "best_bid_qty": 1.5,
                "best_ask": 65021.0, "best_ask_qty": 0.8,
                "bids": [[65020.0, 1.5], [65019.0, 3.0]],  # [price, qty], up to depth 20
                "asks": [[65021.0, 0.8], [65022.0, 2.1]],
            },
            "recent_bars_by_interval": {            # multi-resolution view
                "1m": [...],   # last 60 1m bars (same as recent_bars)
                "5m": [...],   # last 24 5m bars  (downsampled from 1m)
                "1h": [...],   # last 24 1h bars  (downsampled from 1m)
            },
        },
        "ETHUSDT": {...},
        "SOLUSDT": {...},
    },
    "portfolio": {               # all portfolio state is nested here
        "positions": {"BTCUSDT": 0.5},  # only symbols you hold; use .get(sym, 0.0)
        "pnl": {"BTCUSDT": 0.0, "__total": 0.0},
        "cash": 100000.0,            # free cash (float | None)
        "reserved_cash": 0.0,        # cash earmarked by resting buy orders
        "value": 100000.0,           # cash + Σ position·price (None if cash is None)
        "pending_orders": [          # resting limit/stop orders not yet filled/cancelled
            {
                "order_id": "a1b2c3",
                "symbol": "BTCUSDT",
                "action": "buy",
                "quantity": 0.5,
                "order_type": "limit",
                "limit_price": 64000.0,
                "stop_price": None,
                "tif": "gtc",
                "submitted_step": 2,
                "reserved_cash": 32000.0,
                "reserved_position": 0.0,
                "triggered": False,      # stop_limit only: True once the stop has fired
            },
            # ...
        ],
    },
}
```

Notes:
- `price`/`volume`/`timestamp` are `None` only before the first bar
  arrives (cold start); after that they always carry the latest value.
- `order_book` is `None` until the first depth snapshot arrives. Binance
  provides L2 depth (the `bids`/`asks` ladders); use `best_bid`/`best_ask`
  for the touch.
- `pnl["__total"]` is the sum across all symbols and is what drives the
  reward.

## Action format

Return EITHER a single order dict, OR a batch:

```python
# single order
{"action": "buy", "symbol": "BTCUSDT", "quantity": 0.5}

# batch — multiple orders / cancels applied in order, this step
{"orders": [ {<order>}, {<order>}, ... ]}
```

### Order fields

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `action` | `"buy" \| "sell" \| "hold" \| "cancel"` | yes | — | `hold` = do nothing; `cancel` = remove a resting order |
| `symbol` | str | for buy/sell | — | must be one of the input symbols |
| `quantity` | float | for buy/sell | — | must be `> 0` (units of the base asset) |
| `order_type` | `"market" \| "limit" \| "stop" \| "stop_limit"` | no | `"market"` | |
| `limit_price` | float | for limit / stop_limit | — | `> 0`; the price you're willing to trade at |
| `stop_price` | float | for stop / stop_limit | — | `> 0`; the trigger price |
| `tif` | `"ioc" \| "gtc"` | no | `"gtc"` | ignored for `market`/`hold`; `ioc` = fill now or expire, `gtc` = rest until filled or episode end |
| `order_id` | str | for cancel | — | the `order_id` of a resting order from `pending_orders` |

### Order-type semantics

- **market** — fills immediately at the current price (± slippage).
- **limit** — rests until the market reaches `limit_price` (buy fills
  when price ≤ limit; sell fills when price ≥ limit).
- **stop** — rests until the market crosses `stop_price`, then fills at
  market.
- **stop_limit** — rests until `stop_price` triggers, then becomes a
  limit order at `limit_price`.

Resting orders (`gtc` limit/stop/stop_limit) appear in
`observation["pending_orders"]` on subsequent steps until they fill,
you cancel them, or the episode ends.

### Examples

```python
# market buy
{"action": "buy", "symbol": "BTCUSDT", "quantity": 0.5}

# limit buy that rests until price drops to 64000
{"action": "buy", "symbol": "ETHUSDT", "quantity": 2.0,
 "order_type": "limit", "limit_price": 64000.0, "tif": "gtc"}

# stop-loss sell on an existing long position
{"action": "sell", "symbol": "BTCUSDT", "quantity": 0.5,
 "order_type": "stop", "stop_price": 63000.0}

# cancel a resting order seen in pending_orders
{"action": "cancel", "order_id": "a1b2c3"}

# batch: market-buy one symbol and place a take-profit limit on another
{"orders": [
    {"action": "buy", "symbol": "BTCUSDT", "quantity": 0.3},
    {"action": "sell", "symbol": "ETHUSDT", "quantity": 1.0,
     "order_type": "limit", "limit_price": 70000.0},
]}

# do nothing this step
{"action": "hold"}
```

## Order lifecycle & feedback

Orders **never crash the episode** for trading-business reasons —
instead, each step's result is reported back through `info` (which the
runner records). Per step the executor first ticks resting orders
(checking triggers/fills), then applies your new action(s). Outcomes:

- **fills** — orders that executed this step (market fills + any resting
  order that triggered). Each fill carries `order_id`, `action`,
  `symbol`, `quantity`, `market_price`, `executed_price`,
  `slippage_cost`, `transaction_cost`, `order_type`, `status`,
  `filled_step`.
- **accepted** — limit/stop orders that were queued (now resting in
  `pending_orders`).
- **rejections** — orders that failed validation/business checks. Each
  carries `order_intent` (your dict, echoed back), `reason_code`, and a
  human `reason`. Rejection codes: `unknown_symbol`, `invalid_action`,
  `invalid_order_type`, `invalid_tif`, `invalid_quantity`,
  `invalid_price`, `missing_field`, `insufficient_cash`,
  `insufficient_position`, `order_not_found`, `schema_error`.
- **expired** — `ioc` orders that didn't fill immediately, and any
  remaining `gtc` orders cancelled at episode end.

A rejected order does not stop the episode; only an exception raised
inside your `step()` does (recorded as `replay_failure_at`).

## Trading invariants (enforced by `SimulatedExecutor`)

1. **Shorting is allowed, capped by buying power (strict 1×).** A `sell`
   beyond your holdings opens or extends a short (negative position).
2. **Buying power is enforced both ways.** Long and short exposure both
   consume buying power: an order is rejected with `insufficient_cash` if
   it would push gross exposure (`Σ |position|·price` plus cash reserved by
   resting orders) above your equity (`cash + Σ position·price`). Covers
   and position reductions are always allowed.
3. **`quantity > 0`** for `buy`/`sell`; `hold` ignores quantity.
4. **`symbol`** must be one of `BTCUSDT`, `ETHUSDT`, `SOLUSDT`.
5. **Resting buys and resting shorts both reserve buying power** — so a
   pending order's reserved cash is not double-spendable by a later order
   the same step.

## Action frequency — no rate guard

There is **no action-frequency guard** on the in-house `internal_paper`
executor. The gym loop does not pace itself to the bar interval: you may
emit an action every step as fast as the runner can call `step()` —
effectively many actions per second — even though new 1-minute bars only
arrive once per minute. Price freshness is bounded by the live WebSocket
push (sub-second on liquid pairs), not by how often you act. Practically:
acting far faster than the bar cadence means consecutive steps often see
the same latest price, so high-frequency churn mostly just pays slippage.

## Symbols

- **Input:** `["BTCUSDT", "ETHUSDT", "SOLUSDT"]` — every observation
  carries all three.
- **Target:** `["BTCUSDT", "ETHUSDT"]` — only trades on these two
  contribute to PnL / Sharpe / drawdown / win-rate. Trades on `SOLUSDT`
  are accepted (so you can hedge / use it as signal) but are excluded
  from the scored metrics.

## Step budget

`max_steps = 10`. The runner stops after that many `task.step()` calls.
On the final step, any remaining `gtc` orders are cancelled (reported as
`expired`).

## Evaluation

Headline: **`pnl`** (mean per-step PnL, from the curated `PnL` reward;
this is the `reward` key in `reward.json`).

Diagnostics (all in `reward.json`):
- `total_return`, `mean_return`, `std_return`
- `sharpe_ratio`, `max_drawdown`, `win_rate`
- `num_trades`, `num_steps`, `steps_executed`
- `execution_mode_internal_paper = 1.0`
- `replay_failure_at` — present only if your `step()` raised; the index
  of the failing action (PnL then reflects only the prefix that ran).

## Trust model

Your `train.py` controls the action policy via `step(obs)`. Everything
else (provider, executor, gym loop, metric aggregation, `reward.json`
write) runs in code shipped via the openfinai-base image and the bundled
runner — none of it can be tampered with from `train.py` short of
monkey-patching imported modules. The runner instantiates
`RealtimeTradingTask` itself, so the prices your `step` sees are the same
prices the executor marks against.

## Response format

Wrap your code in a fenced markdown block:

```python
# your code here
```

Output ONLY the markdown block — no surrounding prose.
