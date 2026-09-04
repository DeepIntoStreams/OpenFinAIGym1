"""Canonical runtime-context names for generated reward functions.

Ground-truth names are constructor-only; prediction names are forward-only.
Other parameters are literal-default hyperparameters, except the loss name.
Code generation, validation, evaluation, and the reward bank share these sets.
When adding a key, update its set and ensure the evaluator populates it.
"""

from __future__ import annotations


INIT_ONLY_CANONICAL_NAMES: frozenset[str] = frozenset(
    {
        "gt",          # ground-truth tensor for forecasting / pair distances
        "real_emb",    # precomputed real-side embeddings for embedding-pair metrics
        "gt_samples",  # reserved: distributional ground-truth samples [N, S, D]
    }
)

FORWARD_ONLY_CANONICAL_NAMES: frozenset[str] = frozenset(
    {
        "pred",          # agent predictions / generated samples
        "fake_emb",      # precomputed fake-side embeddings
        "pred_samples",  # probabilistic / multi-sample agent forecasts [N, S, D]
    }
)

ALL_CANONICAL_NAMES: frozenset[str] = (
    INIT_ONLY_CANONICAL_NAMES | FORWARD_ONLY_CANONICAL_NAMES
)

# The Loss-base identifier kwarg is special-cased everywhere: it has a default
# of "test" on the base class, it isn't ctx-bound, and it isn't a tunable
# hyperparameter — it just labels the metric instance.
LOSS_IDENTIFIER_KWARG: str = "name"
