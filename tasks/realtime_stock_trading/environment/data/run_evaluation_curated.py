#!/usr/bin/env python3
"""Curated evaluation entry-point for ``realtime_stock_trading``.

Runs entirely inside the agent container — no host-side verifier round-
trip. The actual gym loop, metric aggregation, and reward.json write
live in :func:`openfinai_pipeline.realtime.agent_runtime
.run_realtime_trading_trial` (shipped via openfinai-base).

This bundle uses Alpaca for live market data and the bundled
:class:`SimulatedExecutor` (``execution_mode="internal_paper"``) to
mark portfolio state. No real orders are dispatched. Compare with
``realtime_stock_trading_alpaca_paper`` which uses Alpaca's paper API
as the executor.

Requires ``ALPACA_API_KEY`` + ``ALPACA_SECRET_KEY`` in the container
env (forwarded by ``openfinai_harbor.run_trial`` from the host's
``.env`` automatically when the bundle's class_name starts with
``RealtimeStock``).

The agent's ``train.py`` (in /workspace/) must define::

    def step(observation: dict) -> dict:
        # observation: dict per RealtimeTradingTask.get_observation_space()
        # return: {"action": "buy"|"sell"|"hold", "symbol": str, "quantity": float}
        ...
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="In-container runner for realtime_stock_trading"
    )
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--reward-output", default="/logs/verifier/reward.json")
    args = parser.parse_args()

    try:
        from openfinai_pipeline.realtime.agent_runtime import (
            run_realtime_trading_trial,
        )
        from openfinai_pipeline.realtime.data_providers.alpaca import (
            AlpacaProvider,
        )
    except ImportError as exc:
        sys.stderr.write(
            f"[curated-eval] missing realtime stack in base image: {exc}\n"
            "Rebuild openfinai-base via "
            "`docker build -t openfinai-base:latest -f docker/Dockerfile.base .`\n"
        )
        sys.exit(1)

    # Mirrors task.toml [curated.default_config]. See the same comment
    # in realtime_crypto_trading's runner about why this is duplicated.
    config = {
        "symbols": ["SPY", "QQQ", "IWM"],
        "target_symbols": ["SPY", "QQQ"],
        "slippage_pct": 0.001,
        "transaction_cost_pct": 0.0,
        "max_steps": 10,
        "context_resolutions": [
            {"interval": "1m", "bars": 60},
            {"interval": "5m", "bars": 24},
            {"interval": "1h", "bars": 24},
        ],
        "data_resolution": "1m",
        "execution_mode": "internal_paper",
    }

    # Out-of-market guard: 5× the 1m data_resolution. During regular US
    # equity hours Alpaca's IEX feed produces fresh bars within 1-2
    # minutes; a 5-minute threshold is well past normal jitter but well
    # short of the ~17h gap to the next session, so weekends / holidays /
    # after-hours all trip cleanly without false positives during the
    # trading day. Update this if you change data_resolution above.
    run_realtime_trading_trial(
        provider_factory=AlpacaProvider,
        config=config,
        reward_output_path=args.reward_output,
        max_bar_age_seconds=300.0,
    )


if __name__ == "__main__":
    main()
