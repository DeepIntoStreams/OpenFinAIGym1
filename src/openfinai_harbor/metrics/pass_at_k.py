"""
Pass@k metric for Harbor.

Aggregates rewards from multiple attempts (trials) on the same task,
computing pass@k (best score) and average score for each reward key.
"""
from harbor.metrics.base import BaseMetric


class PassAtK(BaseMetric[dict[str, float | int]]):
    """Aggregate multi-attempt rewards with pass@k (max) and mean."""

    def compute(
        self,
        rewards: list[dict[str, float | int] | None],
    ) -> dict[str, float | int]:
        valid = [r for r in rewards if r is not None]
        if not valid:
            return {"pass_at_k": 0, "mean": 0.0, "n_attempts": len(rewards)}

        all_keys = {k for r in valid for k in r if not k.startswith("_")}

        result: dict[str, float | int] = {
            "n_attempts": len(rewards),
            "n_valid": len(valid),
        }

        for key in sorted(all_keys):
            values = [
                r[key] for r in valid if key in r and isinstance(r[key], (int, float))
            ]
            if not values:
                continue
            result[f"{key}_pass_at_k"] = max(values)
            result[f"{key}_mean"] = round(sum(values) / len(values), 6)

        all_scores = []
        for r in valid:
            task_scores = [
                v
                for k, v in r.items()
                if not k.startswith("_") and isinstance(v, (int, float))
            ]
            if task_scores:
                all_scores.append(sum(task_scores) / len(task_scores))

        if all_scores:
            result["pass_at_k"] = round(max(all_scores), 6)
            result["mean"] = round(sum(all_scores) / len(all_scores), 6)
        else:
            result["pass_at_k"] = 0
            result["mean"] = 0.0

        return result
