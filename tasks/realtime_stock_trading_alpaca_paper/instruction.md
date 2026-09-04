# Realtime Stock Trading via Alpaca Paper API (SPY/QQQ/IWM)

## Problem

Run a real-time trading episode where orders are dispatched to
**Alpaca's paper-trading REST API** (`paper-api.alpaca.markets`), filled
against Alpaca's simulated order book, with PnL read back from the paper
account. The trust handoff: **Alpaca is the oracle** for fills and
account state; the bundled `AlpacaPaperExecutor` (execution mode
`alpaca_paper`) is a thin wrapper around it. Compare with
`realtime_stock_trading`, which uses the in-process `SimulatedExecutor`
against the same Alpaca data feed.

- **Data feed:** Alpaca IEX (free tier) REST + WebSocket.
- **Execution:** real orders to the Alpaca paper account via REST.
  Cash, positions, and fills come from the **live paper account** (the
  framework's `initial_capital` is ignored — the account balance is the
  source of truth). Fills may first appear as `provisional` (booked at
  the submit-time price) and converge to `filled` with the real
  `executed_price` once Alpaca's trade-update push arrives (typically
  <100 ms later).
- **Market:** US regular trading hours only — Alpaca **rejects** orders
  outside RTH (13:30–20:00 UTC / 9:30 AM–4:00 PM ET).
- **Reward:** per-step change in total mark-to-market PnL.

## Required env vars

The agent container needs Alpaca paper credentials:

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

`python -m openfinai_harbor.run_trial` forwards them into the container
automatically for any bundle whose class name starts with
`RealtimeStock`. If they are missing, the trial aborts before docker
spawns with a helpful message.

## What you write

A single Python file at `/workspace/train.py` exposing a top-level
callable:

```python
def step(observation: dict) -> dict:
    """Decide one action (or a batch of orders) for this step.

    Returns
    -------
    dict
        A single order dict, OR a batch {"orders": [<order>, ...]}.
        See "Action format" below.
    """
    ...
```

The bundled runner (`/data/run_evaluation_curated.py`) imports `step` by
name, drives the gym loop with `RealtimeTradingTask` configured for
`execution_mode="alpaca_paper"`, and writes `reward.json`. You do **not**
loop yourself, do **not** call `task.reset()`, do **not** write
`reward.json`.

## Observation shape

The runner serialises everything to plain Python dicts before calling
your `step()`. Bars are plain dicts (`bar["close"]`, not `bar.close`).

```python
{
    "step": 3,                  # steps taken so far
    "steps_remaining": 7,       # -1 if max_steps is unbounded
    "symbols": {
        "SPY": {
            "symbol": "SPY",
            "price": 420.55,                       # latest price (float | None)
            "volume": 1234.0,                       # in-progress bar running volume (float | None)
            "timestamp": "2026-05-09T05:21:00+00:00",  # ISO-8601 of latest bar (str | None)
            "recent_bars": [                        # last 60 1m bars, chronological
                {
                    "symbol": "SPY",
                    "timestamp": "2026-05-09T05:20:00+00:00",
                    "price": 420.40,
                    "open": 420.30, "high": 420.60,
                    "low": 420.25, "close": 420.40,
                    "volume": 1234.0,
                    "extra": {},
                },
                # ...
            ],
            "order_book": {                         # OrderBookSnapshot | None — NBBO only
                "symbol": "SPY",
                "timestamp": "2026-05-09T05:21:00+00:00",
                "best_bid": 420.54, "best_bid_qty": 300.0,
                "best_ask": 420.56, "best_ask_qty": 150.0,
                "bids": [],   # Alpaca exposes NBBO only — no L2 depth ladder
                "asks": [],
            },
            "recent_bars_by_interval": {
                "1m": [...],   # last 60 1m bars (same as recent_bars)
                "5m": [...],   # last 24 5m bars  (downsampled from 1m)
                "1h": [...],   # last 24 1h bars  (downsampled from 1m)
            },
        },
        "QQQ": {...},
        "IWM": {...},
    },
    "portfolio": {               # all portfolio state is nested here
        "positions": {"SPY": 10.0},  # shares held (from the paper account); use .get(sym, 0.0)
        "pnl": {"SPY": 0.0, "__total": 0.0},
        "cash": 100000.0,            # free cash in the paper account (float | None)
        "reserved_cash": 0.0,        # cash earmarked by open buy orders
        "value": 100000.0,           # cash + Σ position·price (None if cash is None)
        "pending_orders": [          # open (un-filled) orders on the paper account
            {
                "order_id": "a1b2c3",
                "symbol": "SPY",
                "action": "buy",
                "quantity": 10.0,
                "order_type": "limit",
                "limit_price": 418.0,
                "stop_price": None,
                "tif": "gtc",
                "submitted_step": 2,
                "reserved_cash": 4180.0,
                "reserved_position": 0.0,
                "triggered": False,
            },
            # ...
        ],
    },
}
```

Notes:
- `cash`/`positions`/`pending_orders` reflect the **live Alpaca paper
  account**, so they include any state the account carried into the
  episode.
- `order_book` is the **NBBO** only — `bids`/`asks` ladders are empty.

## Action format

Return EITHER a single order dict, OR a batch `{"orders": [...]}`. The
schema is identical to `realtime_stock_trading`:

### Order fields

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `action` | `"buy" \| "sell" \| "hold" \| "cancel"` | yes | — | `hold` = do nothing; `cancel` = cancel an open order |
| `symbol` | str | for buy/sell | — | one of the input symbols |
| `quantity` | float | for buy/sell | — | `> 0` (shares) |
| `order_type` | `"market" \| "limit" \| "stop" \| "stop_limit"` | no | `"market"` | mapped 1:1 to the Alpaca order type |
| `limit_price` | float | for limit / stop_limit | — | `> 0` |
| `stop_price` | float | for stop / stop_limit | — | `> 0` |
| `tif` | `"ioc" \| "gtc"` | no | `"gtc"` | mapped to the Alpaca time-in-force |
| `order_id` | str | for cancel | — | `order_id` of an open order from `pending_orders` |

Order-type semantics match a real broker: market fills at the touch;
limit/stop/stop_limit rest on the Alpaca book until triggered, appearing
in `pending_orders` until filled/cancelled/episode-end.

### Examples

```python
# market buy
{"action": "buy", "symbol": "SPY", "quantity": 10.0}

# limit buy resting on the Alpaca book
{"action": "buy", "symbol": "QQQ", "quantity": 5.0,
 "order_type": "limit", "limit_price": 418.0, "tif": "gtc"}

# stop-loss sell
{"action": "sell", "symbol": "SPY", "quantity": 10.0,
 "order_type": "stop", "stop_price": 410.0}

# cancel an open order
{"action": "cancel", "order_id": "a1b2c3"}

# batch
{"orders": [
    {"action": "buy", "symbol": "SPY", "quantity": 5.0},
    {"action": "sell", "symbol": "QQQ", "quantity": 3.0,
     "order_type": "limit", "limit_price": 450.0},
]}

# do nothing
{"action": "hold"}
```

## Order lifecycle & feedback

Per step the executor first polls for newly-resolved orders, then
submits your new action(s). Outcomes are reported through `info`:

- **fills** — orders that filled this step (carrying `order_id`,
  `action`, `symbol`, `quantity`, `market_price`, `executed_price`,
  `order_type`, `status`, `filled_step`). A just-submitted market order
  may show `status="provisional"` until Alpaca confirms the real fill
  price.
- **accepted** — limit/stop orders accepted onto the book (now in
  `pending_orders`).
- **rejections** — orders Alpaca refused (e.g. insufficient buying
  power, order outside RTH, shorting not enabled). Reported in
  `info["rejections"]` with `order_intent`, `reason_code`, `reason` —
  they do **not** crash the episode.
- **expired** — IOC orders that didn't fill, plus open orders cancelled
  at episode end.

Only an exception raised inside your `step()` (or a hard infrastructure
error) stops the episode, recorded as `replay_failure_at`.

## Trading invariants

`AlpacaPaperExecutor` defers to the **Alpaca paper account's** rules
rather than enforcing them in code:

1. **Shorting** is allowed only if your paper account has it enabled;
   otherwise a sell beyond your held position is rejected by Alpaca.
2. **Buying power** is enforced by Alpaca — over-spending is rejected.
3. **`quantity > 0`** for `buy`/`sell`; `hold` ignores quantity.
4. **`symbol`** must be one of `SPY`, `QQQ`, `IWM`.
5. **Orders outside US RTH are rejected** by Alpaca.

## Action frequency — broker rate limits apply

Unlike the in-house `internal_paper` executor (which has no
action-frequency guard), this task sends **real REST orders** to Alpaca,
which enforces an account-level rate limit (~200 requests/min). The
framework does not pace your steps, but it reacts to a `429` by sleeping
per the `Retry-After` header and retrying once. So your *effective*
action rate is bounded by Alpaca's quota and network round-trip, plus
the RTH constraint above — not by a framework-side timer.

## Symbols

- **Input:** `["SPY", "QQQ", "IWM"]` — every observation carries all
  three.
- **Target:** `["SPY", "QQQ"]` — only trades on these two contribute to
  the scored metrics. `IWM` trades are accepted (hedging/signal) but
  excluded from scoring.

## Step budget

`max_steps = 10`. The runner stops after that many `task.step()` calls;
remaining open orders are cancelled at episode end (reported as
`expired`).

## Evaluation

Headline: **`pnl`** (mean per-step PnL; the `reward` key in
`reward.json`).

Diagnostics (all in `reward.json`):
- `total_return`, `mean_return`, `std_return`
- `sharpe_ratio`, `max_drawdown`, `win_rate`
- `num_trades`, `num_steps`, `steps_executed`
- `execution_mode_internal_paper = 0.0` (this task is `alpaca_paper`)
- `replay_failure_at` — present only if your `step()` raised.

## Response format

Wrap your code in a fenced markdown block:

```python
# your code here
```

Output ONLY the markdown block — no surrounding prose.
