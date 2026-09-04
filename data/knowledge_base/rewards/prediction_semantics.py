"""Prediction-array contracts for curated reward classes.

Instruction rendering uses these strings without LLM paraphrasing. Each entry
defines the prediction meaning and any shape or distribution requirement.
Generated rewards fall back to their own docstrings. Add an entry whenever a
curated reward class is introduced.
"""

from __future__ import annotations

# Forecasting Loss classes (RULE A: ``__init__(gt=...)``, ``forward(pred)``).
# ``pred`` and ``gt`` are aligned 1-D (or batch) tensors unless noted.

FORECASTING_LOSS_SEMANTICS: dict[str, str] = {
    "MSELoss": (
        "Each entry of `predictions` is the model's pointwise estimate of "
        "the corresponding scalar target value (same units as `ground_truth`)."
    ),
    "RMSELoss": (
        "Each entry of `predictions` is a pointwise estimate of the "
        "scalar target. Units must match `ground_truth`."
    ),
    "MAELoss": (
        "Each entry of `predictions` is a pointwise estimate of the "
        "scalar target. Units must match `ground_truth`."
    ),
    "MAPELoss": (
        "Each entry of `predictions` is a level (not a return) so the "
        "metric's ratio `(pred - gt) / gt` is meaningful. `ground_truth` "
        "must be non-zero in absolute value."
    ),
    "HuberLoss": (
        "Each entry of `predictions` is a pointwise estimate of the "
        "scalar target. Units must match `ground_truth`."
    ),
    "DirectionalAccuracy": (
        "Each entry of `predictions` is a real-valued signal whose SIGN "
        "is what's scored — the metric counts how often "
        "`sign(predictions[i]) == sign(ground_truth[i])`. Magnitude is "
        "unused."
    ),
    "SignLoss": (
        "Each entry of `predictions` is a real-valued signal scored on "
        "sign agreement with `ground_truth`. Magnitude is unused; only "
        "the sign is read."
    ),
    "QuantileLoss": (
        "Each entry of `predictions` is the model's estimate of the "
        "configured quantile q (default 0.5) of the target distribution at "
        "that index — a level on the same axis as `ground_truth`."
    ),
    "CRPSLoss": (
        "Submit predictive samples via the `pred_samples` ctx kwarg, NOT "
        "via the canonical `predictions` array. `pred_samples` shape is "
        "(N, S) where N matches `ground_truth` and S is the number of "
        "Monte Carlo samples per index. The point-estimate `predictions` "
        "argument is ignored by this metric."
    ),
    "PearsonCorrelation": (
        "Each entry of `predictions` is any cardinal estimate of the "
        "target — only the linear-correlation structure with "
        "`ground_truth` matters; absolute scale and offset cancel."
    ),
    "SpearmanCorrelation": (
        "Each entry of `predictions` is any ordinal estimate — only "
        "the rank order with `ground_truth` matters. Use this when the "
        "paper computes the IC (Information Coefficient)."
    ),
    "F1Score": (
        "Each entry of `predictions` is a real-valued signal; the metric "
        "thresholds at 0 (or the configured `threshold`) and computes "
        "macro-F1 vs. the same threshold applied to `ground_truth`."
    ),
    "R2Score": (
        "Each entry of `predictions` is the model's conditional-mean "
        "estimate of the scalar target. Units must match `ground_truth`."
    ),
    "SharpeRatioLoss": (
        "Each entry of `predictions` is a directional signal; "
        "`sign(predictions[i]) * ground_truth[i]` is treated as a "
        "per-step strategy return and the metric returns the negated "
        "Sharpe ratio of that stream. `ground_truth` must be in return "
        "(not level) units."
    ),
    "SortinoRatio": (
        "Each entry of `predictions` is a directional signal; "
        "`sign(predictions[i]) * ground_truth[i]` is the per-step "
        "strategy return, and the metric returns the negated Sortino "
        "ratio (downside-only deviation in the denominator)."
    ),
}


# Generative Loss classes (sample-level distributional metrics).
# Most consume the agent's batch of generated samples and the real
# reference batch directly via ``pred`` / ``gt`` (RULE A pattern). The
# embedding-pair metrics expect pre-computed embeddings instead.

GENERATIVE_LOSS_SEMANTICS: dict[str, str] = {
    "CovLoss": (
        "`predictions` is a batch of generated time-series samples shaped "
        "[B, T, D] matching `ground_truth`. The metric compares "
        "feature-cross-section covariance structure between the two."
    ),
    "ACFLoss": (
        "`predictions` is a batch of generated samples [B, T, D]; the "
        "metric compares per-channel autocorrelation up to `max_lag` "
        "against `ground_truth`."
    ),
    "CrossCorrelLoss": (
        "`predictions` is a batch of generated samples [B, T, D]; the "
        "metric compares lagged cross-correlations (between channels and "
        "across time) against `ground_truth`."
    ),
    "MeanLoss": (
        "`predictions` is a batch of generated samples; the metric "
        "computes the L1 distance between the marginal mean of "
        "`predictions` and that of `ground_truth`."
    ),
    "HistogramLoss": (
        "`predictions` is a batch of generated samples [B, T, D]; per "
        "time-step the metric compares the marginal histogram of "
        "`predictions` against `ground_truth`."
    ),
    "Sig_MMD_loss": (
        "`predictions` is a batch of generated path samples; the metric "
        "is the path-signature MMD against the real reference batch in "
        "`ground_truth`."
    ),
    "SigW1Loss": (
        "`predictions` is a batch of generated path samples; the metric "
        "is the W1 distance on truncated path signatures vs. "
        "`ground_truth`."
    ),
    "ONNDMetricLoss": (
        "`predictions` is a batch of generated samples; the metric "
        "computes outgoing nearest-neighbour distance against the real "
        "reference (proxy for diversity / coverage)."
    ),
    "ICDMetricLoss": (
        "`predictions` is a batch of generated samples; the metric "
        "computes intra-class (within-batch) pairwise distance — a "
        "diversity proxy."
    ),
    "VARMetricLoss": (
        "`predictions` is a batch of generated samples; the metric "
        "compares the alpha-VaR or expected-shortfall tail statistic of "
        "`predictions` against `ground_truth` (configure `alpha` and "
        "`statistic` via per-task params)."
    ),
    "EmbeddingFID": (
        "Submit pre-computed sample embeddings as the `fake_emb` ctx "
        "kwarg; the real reference embeddings come in via `real_emb`. "
        "The canonical `predictions` array is NOT consumed by this "
        "metric — supply the embedding tensors instead. The task's "
        "``prepare_eval_context`` hook is the typical site to populate "
        "these."
    ),
    "EmbeddingKID": (
        "Submit pre-computed sample embeddings as the `fake_emb` ctx "
        "kwarg; `real_emb` carries the real reference embeddings. The "
        "canonical `predictions` array is NOT consumed by this metric — "
        "use the task's ``prepare_eval_context`` hook to fill the "
        "embedding kwargs."
    ),
}


# TradingReward classes (per-trade dict[str, Any] streams).
# Auto-generated benchmark tasks DO NOT emit trading interaction models,
# so these entries are mainly here for completeness when a curated trading
# task shares the bank.

TRADING_REWARD_SEMANTICS: dict[str, str] = {
    "DirectionAccuracy": (
        "Each entry of `predictions` is a dict with `direction` "
        "(`'long'` / `'short'` / `'flat'`); `ground_truths[i]` carries "
        "`actual_return`. Score is 1.0 when the signed direction matches "
        "the sign of `actual_return`."
    ),
    "PnL": (
        "Each entry of `predictions` is a dict with `direction` and "
        "`quantity`; `ground_truths[i]` provides `actual_return`. The "
        "metric sums signed unit-PnL across the trade list."
    ),
    "ReturnMAE": (
        "Each entry of `predictions` is a dict whose `direction` and "
        "`quantity` define an implied per-trade return; the metric "
        "averages the absolute error against `ground_truths[i]"
        "['actual_return']`."
    ),
    "SharpeRatio": (
        "Per-trade dicts with `direction` and `quantity`; the metric "
        "computes mean(per-trade-PnL) / std(per-trade-PnL) — annualisation "
        "free."
    ),
    "MaxDrawdown": (
        "Per-trade dicts with `direction` and `quantity`; the metric "
        "returns the worst peak-to-trough drawdown in cumulative PnL."
    ),
    "WinRate": (
        "Per-trade dicts with `direction` and `quantity`; the metric "
        "returns the fraction of trades with positive realised PnL."
    ),
    "QuantityPnL": (
        "Each entry of `predictions` is a dict with `direction`, "
        "`quantity`, optional `entry_price`/`exit_price`, plus optional "
        "`slippage`/`fee` overrides; the metric sums the explicit "
        "quantity-aware PnL across trades."
    ),
}


# Public lookup

_ALL_SEMANTICS: dict[str, str] = {
    **FORECASTING_LOSS_SEMANTICS,
    **GENERATIVE_LOSS_SEMANTICS,
    **TRADING_REWARD_SEMANTICS,
}


def get_prediction_semantics(class_name: str) -> str:
    """Return the curated prediction-contract sentence(s) for a metric class.

    Returns an empty string for classes not in the curated bank (typically
    Phase-3-generated rewards, whose semantics live in their own docstring
    inside the per-task ``task_rewards.py``).
    """
    return _ALL_SEMANTICS.get(class_name, "")


__all__ = [
    "FORECASTING_LOSS_SEMANTICS",
    "GENERATIVE_LOSS_SEMANTICS",
    "TRADING_REWARD_SEMANTICS",
    "get_prediction_semantics",
]
