#!/usr/bin/env python3
"""Curated evaluation entry-point for ``realtime_crypto_trading``.

Runs entirely inside the agent container — no host-side verifier round-
trip. The actual gym loop, metric aggregation, and reward.json write
live in :func:`openfinai_pipeline.realtime.agent_runtime
.run_realtime_trading_trial` (shipped via openfinai-base). This file
just supplies the bundle-specific provider factory + default config.

The agent's ``train.py`` (in /workspace/) must define::

    def step(observation: dict) -> dict:
        # observation: dict per RealtimeTradingTask.get_observation_space()
        # return: {"action": "buy"|"sell"|"hold", "symbol": str, "quantity": float}
        ...

Failure modes (missing train.py, missing ``step``, executor-invariant
violation, network failure) all surface as a numeric-clean reward.json
+ sibling error.txt — never an unhandled traceback.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="In-container runner for realtime_crypto_trading"
    )
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--reward-output", default="/logs/verifier/reward.json")
    args = parser.parse_args()

    # Local imports so the runner can fail soft if the openfinai-base
    # image is missing the realtime stack — the error message points at
    # the right rebuild command.
    try:
        from openfinai_pipeline.realtime.agent_runtime import (
            run_realtime_trading_trial,
        )
        from openfinai_pipeline.realtime.data_providers.binance import (
            BinanceProvider,
        )
    except ImportError as exc:
        sys.stderr.write(
            f"[curated-eval] missing realtime stack in base image: {exc}\n"
            "Rebuild openfinai-base via "
            "`docker build -t openfinai-base:latest -f docker/Dockerfile.base .`\n"
        )
        sys.exit(1)

    # Mirrors task.toml [curated.default_config]. Kept inline here so the
    # bundle's runtime configuration is self-contained inside the
    # container — no need to ship task.toml or parse it. Update both
    # places when changing defaults; the verifier handler is disabled
    # for trading bundles so task.toml's config doesn't actually drive
    # any host-side execution path.
    config = {
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "target_symbols": ["BTCUSDT", "ETHUSDT"],
        "slippage_pct": 0.001,
        "transaction_cost_pct": 0.0,
        "max_steps": 10,
        "context_resolutions": [
            {"interval": "1m", "bars": 60},
            {"interval": "5m", "bars": 24},
            {"interval": "1h", "bars": 24},
        ],
        "data_resolution": "1m",
        # Binance has no paper-trading API — internal_paper is the only
        # viable mode for crypto. The literal lives in TradingConfig's
        # default but is repeated here for clarity.
        "execution_mode": "internal_paper",
    }

    run_realtime_trading_trial(
        provider_factory=BinanceProvider,
        config=config,
        reward_output_path=args.reward_output,
    )


if __name__ == "__main__":
    main()
