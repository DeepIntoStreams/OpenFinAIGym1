"""Single source of truth for the reward_fn contract (R1-R6).

Both the codegen prompt (``benchmark/templates/reward_fn_prompt.j2``) and the
reviewer prompt (``corpus_builder.build_reward_review_prompt``) include this
rule list verbatim, so the two sides cannot drift. The AST validator
(``corpus.contract_validator``) enforces the same rules statically.

The principle: parameter names ARE the wiring. A no-default param whose
name is in the canonical ctx-key set binds to ``ctx[<name>]`` at runtime;
anything else with a literal default is a tunable hyperparameter; anything
else is a contract violation. No JSON envelope is required.
"""

from openfinai_pipeline.prompts.canonical_ctx import (
    FORWARD_ONLY_CANONICAL_NAMES,
    INIT_ONLY_CANONICAL_NAMES,
    LOSS_IDENTIFIER_KWARG,
)


def _format_name_set(names: frozenset[str]) -> str:
    return ", ".join(sorted(names))


REWARD_FN_CONTRACT_RULES: str = f"""\
=== REWARD_FN CONTRACT (RULES R1-R6) ===

These rules are the canonical reward_fn contract. They are enforced by an
AST pre-flight check before pytest runs, by the unified pytest harness
under tests/test_reward_bank.py, and by the reviewer in subsequent rounds.
Violating any rule means the round is rejected before pytest even runs.

CANONICAL CTX-KEY SET
  Init-only (captured at __init__):
    {_format_name_set(INIT_ONLY_CANONICAL_NAMES)}
  Forward-only (consumed at forward()):
    {_format_name_set(FORWARD_ONLY_CANONICAL_NAMES)}

  These names are the wiring. A parameter named ``gt`` IS bound to
  ``ctx["gt"]`` at runtime; you do NOT declare it anywhere else. Same for
  every other canonical name.

R1 (canonical name => no default).
  A parameter whose name is in the canonical set MUST NOT have a default
  value. These params carry essential ctx tensors and are always required.
    # WRONG:
    def __init__(self, gt=None, name="test"): ...
    # CORRECT:
    def __init__(self, gt, name="test", alpha=0.5): ...

R2 (non-canonical name => must have default).
  Any parameter whose name is NOT in the canonical set (and is not the
  Loss-base ``{LOSS_IDENTIFIER_KWARG}`` identifier kwarg) MUST have a
  literal default value. These are hyperparameters: alpha, eps, depth,
  quantile, lookback, etc. If you need a per-task constant the dataset
  provides, give it a sensible default for now and flag the metric.
    # WRONG (can't be bound at runtime):
    def __init__(self, gt, lookback): ...
    # CORRECT:
    def __init__(self, gt, lookback=20): ...

R3 (init-only canonical names forbidden in forward()).
  forward() MUST NOT take any parameter whose name is in
  {{{_format_name_set(INIT_ONLY_CANONICAL_NAMES)}}}.
  The agent must not see ground truth at scoring time; capture it in
  ``__init__`` instead.
    # WRONG (RULE R3 violation — agent would peek at gt):
    def forward(self, pred, gt): ...
    # CORRECT:
    def __init__(self, gt): self.gt = gt
    def forward(self, pred): ...

R4 (forward-only canonical names forbidden in __init__).
  __init__ MUST NOT take any parameter whose name is in
  {{{_format_name_set(FORWARD_ONLY_CANONICAL_NAMES)}}}.
  The evaluator caches the metric instance after first ``score()`` call, so
  a pred bound at init would be frozen on the first call's data.
    # WRONG:
    def __init__(self, pred): self.pred_first = pred  # frozen forever
    # CORRECT:
    def forward(self, pred): ...

R5 (one Loss subclass).
  Define EXACTLY ONE Loss subclass per file. Helper classes (no Loss base)
  are fine, but only one of them may inherit from Loss.

R6 (torch only, no separate compute).
  All ops in torch (no numpy at runtime). forward() does its own reduction
  and returns a scalar torch.Tensor. Do NOT define a separate compute()
  method — the base Loss class has no forward->compute() wrapper.

NAME PRECISION (Layer 1)
  Param names should describe WHAT the tensor semantically is, not the
  role it plays. Generic names like ``output``, ``value``, ``result``,
  ``data``, ``important_output`` are forbidden — the reviewer will reject
  them. Pick a name from the canonical set if one fits; if you need
  something new (e.g. ``volatility_forecast``, ``regime_label``,
  ``classification_logits``), name it after the upstream semantic role,
  not its position in your formula.

INPUT SEMANTICS (Layer 1)
  For classification-style metrics (ECE, Brier, calibration, etc.) assume
  ``pred`` is already in probability form (binary: scalar in [0, 1];
  multiclass: normalized across the last axis); do NOT apply
  softmax/sigmoid inside the metric.
"""
