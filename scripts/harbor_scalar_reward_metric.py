# /// script
# dependencies = []
# ///
"""Aggregate Harbor verifier rewards using a scalar reward convention.

Harbor's built-in mean/min/max/sum metrics expect each verifier reward to be a
single-key dictionary. OpenFinGym tasks often return rich metric dictionaries
such as {"mse": ..., "total_loss": ..., "reward": ...}. This adapter keeps
those rich per-trial metrics intact while giving Harbor a single scalar for
job-level aggregation.
"""

import argparse
import json
from numbers import Real
from pathlib import Path
from typing import Any


PREFERRED_SCALAR_KEYS = ("reward", "score")


def _as_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be numeric, got {type(value).__name__}")
    return float(value)


def scalarize_reward(reward: Any, *, line_number: int) -> float:
    """Convert one verifier reward object into one scalar.

    Priority:
    1. Use a standard scalar field such as "reward" or "score".
    2. If the reward has exactly one numeric key, use that value.
    3. As an OpenFinGym loss-task fallback, convert non-negative "total_loss" to
       1 / (1 + total_loss).
    4. Otherwise fail loudly with a helpful message.
    """
    if reward is None:
        return 0.0

    if isinstance(reward, Real) and not isinstance(reward, bool):
        return float(reward)

    if not isinstance(reward, dict):
        raise ValueError(
            f"Reward on line {line_number} must be an object, number, or null; "
            f"got {type(reward).__name__}"
        )

    for key in PREFERRED_SCALAR_KEYS:
        if key in reward:
            return _as_float(reward[key], context=f"Reward field {key!r} on line {line_number}")

    numeric_items = [
        (key, value)
        for key, value in reward.items()
        if isinstance(value, Real) and not isinstance(value, bool)
    ]
    if len(numeric_items) == 1:
        return float(numeric_items[0][1])

    if "total_loss" in reward:
        total_loss = _as_float(
            reward["total_loss"],
            context=f"Reward field 'total_loss' on line {line_number}",
        )
        if total_loss < 0:
            raise ValueError(
                f"Reward field 'total_loss' on line {line_number} must be non-negative "
                "to derive a scalar reward"
            )
        return 1.0 / (1.0 + total_loss)

    raise ValueError(
        f"Could not scalarize reward on line {line_number}. "
        f"Expected one of {PREFERRED_SCALAR_KEYS}, exactly one numeric key, "
        f"or non-negative 'total_loss'. Keys: {sorted(reward.keys())}"
    )


def main(input_path: Path, output_path: Path) -> None:
    values: list[float] = []

    for line_number, line in enumerate(input_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        values.append(scalarize_reward(json.loads(line), line_number=line_number))

    mean_reward = sum(values) / len(values) if values else 0.0
    output_path.write_text(json.dumps({"mean_reward": mean_reward}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        required=True,
        help="Path to a jsonl file containing verifier rewards.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        required=True,
        help="Path to write the aggregate metric JSON object.",
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path)
