"""Normalize curated score mappings and select a headline reward.

An explicit reward takes precedence, followed by the configured headline
metric. Missing or NaN headlines become zero; private keys are omitted.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def coerce_scores(
    scores: Any, headline_metric: str
) -> Tuple[Dict[str, float], float]:
    """Flatten ``scores`` to ``(flat_dict, weighted_total)``.

    See module docstring for the headline-metric resolution rules.
    """
    raw_scores = dict(scores) if isinstance(scores, dict) else {}
    explicit_reward: Optional[float] = None
    flat: Dict[str, float] = {}
    for name, value in raw_scores.items():
        if name.startswith("_"):
            continue
        if name == "reward":
            try:
                explicit_reward = float(value)
            except (TypeError, ValueError):
                explicit_reward = None
            continue
        try:
            flat[name] = float(value)
        except (TypeError, ValueError):
            flat[name] = float("nan")

    if explicit_reward is not None:
        weighted_total = explicit_reward
    elif headline_metric in flat:
        candidate = flat[headline_metric]
        # NaN-safe: ``x == x`` is False iff x is NaN.
        weighted_total = candidate if candidate == candidate else 0.0
    else:
        weighted_total = 0.0
    return flat, weighted_total


__all__ = ["coerce_scores"]
