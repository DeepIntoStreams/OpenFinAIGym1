#!/usr/bin/env python3
"""Curated evaluation entry-point for ``realtime_stock_trading_alpaca_paper``.

Same pattern as ``realtime_stock_trading`` but with
``execution_mode="alpaca_paper"`` — orders are dispatched to Alpaca's
paper-trading REST endpoint (``paper-api.alpaca.markets``) instead of
being marked against an in-process simulator. The agent's actions
become real (paper) orders against a virtual account; the bundled
:class:`AlpacaPaperExecutor` handles dispatch, fills, and position
queries. Metrics still go through the same reward bank.

Requires ``ALPACA_API_KEY`` + ``ALPACA_SECRET_KEY`` in the container
env (forwarded by ``openfinai_harbor.run_trial``).

The agent's ``train.py`` (in /workspace/) must define::

    def step(observation: dict) -> dict:
        ...
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "In-container runner for realtime_stock_trading_alpaca_paper"
        )
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

    # Mirrors task.toml [curated.default_config]. execution_mode is
    # hardcoded to alpaca_paper — orders go to paper-api.alpaca.markets.
    config = {
        "symbols": ["SPY", "QQQ", "IWM"],
        "target_symbols": ["SPY", "QQQ"],
        "max_steps": 10,
        "context_resolutions": [
            {"interval": "1m", "bars": 60},
            {"interval": "5m", "bars": 24},
            {"interval": "1h", "bars": 24},
        ],
        "data_resolution": "1m",
        "execution_mode": "alpaca_paper",
    }

    # Out-of-market guard: see realtime_stock_trading's runner for the
    # threshold rationale. The alpaca_paper executor would fail loudly
    # mid-loop on a closed market (paper-api rejects orders) — this
    # guard short-circuits to a clean reward.json before that happens.
    run_realtime_trading_trial(
        provider_factory=AlpacaProvider,
        config=config,
        reward_output_path=args.reward_output,
        max_bar_age_seconds=300.0,
    )


if __name__ == "__main__":
    main()
