"""Per-dataset ``load.py`` codegen + reviewer prompts.

The loader's target/horizon/leakage decisions depend on what the SPECIFIC
TASK is asking the agent to predict, so loader generation lives in Phase
4 where the paper text, ML task summary, experiments, and reward set are
available. This module owns the LLM-side prompts; the round-loop driver
lives in :mod:`openfinai_pipeline.benchmark.loader.executor`.
"""

from __future__ import annotations

import json
from typing import Any

from openfinai_pipeline.prompts.corpus_builder import _truncate, review_schema

__all__ = [
    "build_write_loader_code_prompt",
    "build_loader_review_prompt",
    "loader_code_generation_schema",
    # re-export for callers
    "review_schema",
]


def _format_rewards_block(rewards: list[dict[str, Any]] | None) -> str:
    """Render the task's reward list as a short bullet block for the prompt.

    Each entry is ``- {name}: {description}`` (description truncated to a
    short summary so the prompt isn't dominated by reward prose). The
    LLM uses this to understand what scoring the loader's
    ``ground_truth`` will feed.
    """
    if not rewards:
        return "(none)"
    lines: list[str] = []
    for reward in rewards:
        if not isinstance(reward, dict):
            continue
        name = str(reward.get("name", "")).strip()
        if not name:
            continue
        description = str(reward.get("description", "")).strip()
        if description:
            description = description[:400]
            lines.append(f"- {name}: {description}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines) if lines else "(none)"


def _task_context_block(
    *,
    task_title: str,
    ml_task_summary: str,
    experiments: str,
    rewards: list[dict[str, Any]] | None,
    paper_context: str,
) -> str:
    """Render the task-level context block shared by both loader prompts."""
    title_block = (task_title or "").strip() or "(none)"
    summary_block = (
        _truncate((ml_task_summary or "").strip(), field_name="ml_task_summary")
        or "(none)"
    )
    experiments_block = (
        _truncate((experiments or "").strip(), field_name="experiments")
        or "(none)"
    )
    rewards_block = _format_rewards_block(rewards)
    paper_block = (
        _truncate(
            (paper_context or "").strip(),
            max_chars=60_000,
            field_name="paper_context",
        )
        or "(none)"
    )
    return (
        "Task this loader serves\n"
        f"- Title: {title_block}\n"
        f"- ML task summary:\n{summary_block}\n"
        f"- Experiments / training horizon:\n{experiments_block}\n"
        f"- Rewards (these score the agent's output against ground_truth):\n"
        f"{rewards_block}\n"
        f"- Paper context (excerpt):\n{paper_block}\n"
    )


def build_write_loader_code_prompt(
    *,
    name: str,
    description: str,
    downloaded_dataset_description: str,
    interaction_model: str,
    artifacts: list[dict],
    preview: Any,
    labeler_notes: str,
    task_title: str = "",
    ml_task_summary: str = "",
    experiments: str = "",
    rewards: list[dict[str, Any]] | None = None,
    paper_context: str = "",
    previous_review: str = "",
    previous_script: str = "",
    execution_log: str = "",
) -> str:
    """Build the per-dataset ``load.py`` codegen prompt.

    The loader is generated AFTER labeling AND with full task context: the
    paper, the task summary, the experiments, and the reward set. The LLM
    uses both layers to decide which column is the target, whether to
    apply a horizon shift, and which feature columns to drop for leakage.

    Phase 4 will import the produced ``load.py`` to obtain
    ``(features, ground_truth)`` numpy arrays at evaluation time.

    The artifact list (``role`` per file, plus columns/preview) is the
    only signal the LLM gets about what's on disk; it judges the
    ground_truth source itself rather than being routed by a
    labeler-authored ``has_ground_truth`` boolean. This keeps the
    contract outcome-based: the LLM is free to load a separate GT
    artifact, read a precomputed target column from a feature file, or
    derive a target via transform — whichever fits the task best.
    """
    desc_block = (
        _truncate((description or "").strip(), field_name="description") or "(none)"
    )
    downloaded_desc_block = (
        _truncate(
            (downloaded_dataset_description or "").strip(),
            field_name="downloaded_dataset_description",
        )
        or "(none)"
    )
    artifacts_block = (
        _truncate(
            json.dumps(list(artifacts or []), ensure_ascii=False, indent=2),
            field_name="artifacts",
        )
        or "[]"
    )
    preview_block = (
        _truncate(
            json.dumps(preview or {}, ensure_ascii=False, indent=2),
            field_name="preview",
        )
        or "{}"
    )
    labeler_notes_block = (
        _truncate((labeler_notes or "").strip(), field_name="labeler_notes")
        or "(none)"
    )
    previous_script_block = (
        _truncate(previous_script.strip(), field_name="previous_script") or "(none)"
    )
    execution_log_block = (
        _truncate(
            execution_log.strip(), keep_tail=True, field_name="execution_log"
        )
        or "(none)"
    )
    previous_review_block = (
        _truncate(previous_review.strip(), field_name="previous_review") or "(none)"
    )
    interaction_block = (interaction_model or "").strip().lower() or "(none)"
    task_block = _task_context_block(
        task_title=task_title,
        ml_task_summary=ml_task_summary,
        experiments=experiments,
        rewards=rewards,
        paper_context=paper_context,
    )
    return f"""
You are writing a small dataset loader script. The dataset has just been
downloaded and labeled. Your loader will be imported by the Phase 4
evaluator and called as ``load(data_dir)`` whenever the evaluator needs
ground-truth values for scoring. The loader is task-aware: the same
underlying dataset may serve multiple tasks with different prediction
targets and horizons, so use the task context below to pick the right
target column.

{task_block}

Dataset identity
- Name: {name}
- Paper description: {desc_block}
- Downloaded-dataset description: {downloaded_desc_block}

Labeling decisions (manifest)
- interaction_model: {interaction_block}
- artifacts (role + format + columns + shape + dtype, one per file):
{artifacts_block}
- labeler_notes (prose hints from the labeler — may name a target column
  or warn about format quirks; treat as advisory, not authoritative):
{labeler_notes_block}

Concrete file previews on disk
{preview_block}

Previous round script (empty if first round)
{previous_script_block}

Previous round execution log (pytest output)
{execution_log_block}

Previous round reviewer feedback (empty if first round)
{previous_review_block}

# Loader contract — you MUST satisfy this exactly

Write a Python module named ``load.py`` containing exactly one public
function:

```python
from pathlib import Path
import numpy as np

def load(data_dir: Path) -> dict:
    \"\"\"Load and pre-split this dataset's features + ground_truth.

    Args:
        data_dir: Path to the directory holding the files listed in the
                  manifest's ``artifacts`` array (relative paths).

    Returns:
        For forecasting and conditional-generative tasks (the common
        case), the **B-shape** dict:

            {{
                "train": {{"features": ndarray, "ground_truth": ndarray}},
                "test":  {{"features": ndarray, "ground_truth": ndarray}},
                "split_policy": str,
                "split_metadata": dict | None,
            }}

        For unconditional generative tasks where there is no held-out
        target (distributional metrics over a single reference set),
        the **reference-shape** dict:

            {{
                "reference": ndarray,
                "split_policy": "no_split",
                "split_metadata": None,
            }}

        ``features`` and ``ground_truth`` may be either single ndarrays
        (univariate / single-symbol case) or ``{{symbol: ndarray}}``
        dicts (multi-symbol panel case). All ndarray leaves MUST be
        float dtype, finite, non-empty.
    \"\"\"
```

# Hard rules

1. **Pick a shape.** B-shape (``"train"``/``"test"`` keys) is the
   default for forecasting and conditional-generative. Reference-shape
   (``"reference"`` key) is reserved for unconditional generative
   tasks where the paper does NOT define a held-out evaluation set
   (distributional metrics like FID over a single reference). Mixing
   the two (e.g. having BOTH ``train`` and ``reference``) is invalid.

2. **Pick a ``split_policy``** value:
   * ``"chronological_80_20"`` — the dataset is time-ordered; train is
     the first 80% of rows, test is the last 20%. Default for time
     series with no paper-specified split.
   * ``"random_80_20"`` — the dataset is i.i.d. (no time order);
     shuffle with a fixed seed and take 80/20.
   * ``"paper_dates"`` — the paper specifies explicit date boundaries
     (e.g. train=2009-2018, test=2021-2023). Apply them yourself and
     populate ``split_metadata`` with the dates you used.
   * ``"no_split"`` — only valid for reference-shape (unconditional
     generative). Forbidden for B-shape.

3. **``split_metadata``** is a small dict you populate with the
   structured boundary information you derived. Common keys:
   ``train_end``, ``test_start``, ``train_dates``, ``test_dates``,
   ``random_seed``, ``rows_train``, ``rows_test``. Set to ``None``
   when not applicable.

4. **B-shape: ``train["ground_truth"]`` and ``test["ground_truth"]``
   MUST both be non-None float ndarrays.** The install slicer omits
   ``test["ground_truth"]`` from the agent-readable bundle and writes
   it to a verifier-only mount; what YOU emit at the loader layer
   must contain BOTH so the slicer can demux. ``train["features"]``
   and ``test["features"]`` MUST also be non-None.

5. **Per-split alignment.** For forecasting tasks:
   ``len(train["features"]) == len(train["ground_truth"])`` AND
   ``len(test["features"]) == len(test["ground_truth"])``. For panel
   data (dict leaves) the alignment must hold per symbol.

6. **Source the ground_truth fit-for-purpose.** Pick whatever
   produces a ``ground_truth`` array that the task's rewards will
   actually score against the paper's stated target. You have full
   discretion over the mechanism:

   * Read a separate GT artifact if the labeler tagged one
     (``role="ground_truth"`` or ``role="reference"``).
   * Read a precomputed target column already present in a feature
     file (downloaders sometimes pre-shift returns / labels into the
     same CSV — e.g. ``target_return_20d``, ``label``).
   * Derive a target by transform of feature columns (e.g.
     ``close.shift(-h) / close - 1`` for an h-step forward return).
   * Combine — load a base column from one artifact and apply a
     horizon transform if needed.

   No mechanism is privileged. The reviewer judges whether your
   chosen ``ground_truth`` matches the task's prediction objective
   and the rewards' input shape, not which path you took. Whatever
   path you pick, fill in ``ground_truth_provenance`` (see Output
   below) so the choice is auditable.

7. **No target leakage in features.** Whatever ``ground_truth``
   you produce, the corresponding ``features`` array MUST NOT
   contain literal copies, affine transforms, or named derivatives
   of the target at the same time index. Specifically EXCLUDE:
   (a) The target column itself (literal name match).
   (b) Any column whose name CONTAINS the target column name as a
       case-insensitive substring (e.g. target=``close`` → drop
       ``scaled_close``, ``pctdiff_close``, ``log_close``,
       ``norm_close``, ``next_day_close``, ``close_pct_chg``).
   (c) Any column the description/preview documents as a min-max-
       scaled, z-score-normalized, log, percentage-difference, or
       signed-return variant of the target.
   (d) Technical indicators computed solely from the target column
       (``MACD_close``, ``RSI_close``, ``EMA_close``, ...).
   Do NOT rely on dropping only the literal target column name;
   derived columns (scaled, log, pct-diff) are also leakage.

8. **Same-day OHLC vs same-day close.** If your target is the
   same-day close and your features include same-day
   open/high/low at the same row index, ``low[t] <= close[t] <= high[t]``
   makes high/low partial reveals of the target. Either:
   (a) lag the OHLC features by one row before alignment (predict
       ``close[t]`` from ``OHLC[t-1]``), or
   (b) drop high and low (keep open/volume only).
   If your target is already forward-shifted (``next_day_close``,
   ``return_t+1``, ``target_return_20d``), same-row OHLC is fine —
   just record that fact in ``ground_truth_provenance.leakage_audit``.

9. **Chronological order.** When ``split_policy="chronological_80_20"``
   or ``"paper_dates"``, every row in ``train`` MUST precede every
   row in ``test`` along the time axis. For panel data, this holds
   per-symbol. Sort by date BEFORE splitting.

10. Drop non-numeric columns (dates, IDs, free-text) from features
    by default. Use pandas to read messy CSVs
    (``pd.read_csv(path)``); only use ``np.loadtxt`` if you have
    already proved the file is purely numeric. Coerce to
    ``float64`` and drop or fill NaNs before returning.

11. NO top-level work outside ``load()``. No I/O, no heavy imports
    (torch, sklearn, etc.) at module scope — keep imports inside
    the function or limited to ``import numpy as np``,
    ``from pathlib import Path``, and the parsing libs you
    actually need.

12. Do not depend on the working directory; resolve all paths
    relative to ``data_dir``.

13. The script must compile cleanly and pass the loader pytest
    contract (clean import, correct dict shape, finite arrays,
    length alignment for forecasting, no obvious feature→target
    leakage).

# Output

Respond with JSON containing exactly four fields, in this order:

- ``derivation_rationale``: a short prose block (typically 5-10
  sentences) that walks through, in order: (a) **what you're
  predicting**, in plain English, drawn from the task summary /
  experiments / paper / rewards (e.g. "20-day forward return per
  stock per day, scored cross-sectionally by IC against realised
  20-day returns"); (b) **shape + split** — which output shape
  (B-shape or reference-shape), which ``split_policy``, and how
  you derived the per-split row counts; for ``paper_dates``, name
  the explicit dates; for ``chronological_80_20``, confirm rows
  are sorted by date before slicing.
  Write this BEFORE the structured fields so your decisions are
  committed in tokens before the code lands.

- ``ground_truth_provenance``: a structured record describing
  exactly where ``ground_truth`` came from. Fields:
    * ``source``: one of ``"separate_artifact"``,
      ``"precomputed_column"``, or ``"derived_transform"``.
    * ``column_or_artifact``: the column name (for
      ``precomputed_column`` / ``derived_transform``) or the
      artifact path (for ``separate_artifact``). Use the literal
      name verbatim — the reviewer and the leakage scan use it
      as an anchor.
    * ``transform``: a one-line Python-flavoured expression
      describing the derivation when ``source="derived_transform"``
      (e.g. ``"close.shift(-20) / close - 1"``); ``null`` for the
      other two sources.
    * ``horizon_units``: human-readable horizon string when
      relevant (``"20 trading days"``, ``"1 minute"``); ``null``
      if the target is contemporaneous or the concept does not
      apply.
    * ``paper_target_quote``: a short, verbatim or near-verbatim
      quote from the paper / experiments / rewards that defines
      the target you chose. The reviewer uses this to confirm
      that what you produced matches what the paper actually
      scores. If you cannot find a clear quote, say so —
      including a stretched or invented justification will be
      treated as a red flag.
    * ``leakage_audit``: a one-sentence statement listing the
      feature columns you dropped (or kept) for leakage
      prevention and why. If nothing needed dropping, say so
      explicitly. This replaces the old "rule 13" prose
      requirement.

- ``code``: the full Python source of ``load.py``, implementing
  the choices declared above.

- ``split_policy``: the structured policy string. One of:
  ``"chronological_80_20"``, ``"random_80_20"``, ``"paper_dates"``,
  ``"no_split"`` (only with reference-shape). Emit this AFTER
  ``code``: it's a label describing what the loader you just
  wrote actually returns, so it must match the ``"split_policy"``
  key in the returned dict. The slicer reads this alongside the
  data and writes it as an attribute on the agent-readable
  ``dataset.h5``.
""".strip()


def loader_code_generation_schema() -> dict:
    """JSON schema for the per-dataset loader codegen LLM output.

    Field order is load-bearing for OpenAI-style structured outputs:
    ``derivation_rationale`` comes first (high-level task framing),
    then ``ground_truth_provenance`` (the structured commitment to
    where GT comes from — written before ``code`` so the LLM can't
    drift in code emission), then ``code``, then ``split_policy``
    (a label describing what the emitted code actually returns).

    ``ground_truth_provenance`` is the auditable record of which
    mechanism the loader used to source GT (separate artifact /
    precomputed column / derived transform). The reviewer judges
    fit-for-purpose against ``paper_target_quote`` and the rewards;
    no mechanism is privileged. Downstream consumers (manifest,
    slicer, evaluator) can also read it.

    ``split_policy`` is the structured discriminator the install
    slicer reads to know whether the LLM emitted B-shape or
    reference-shape and what split rule was applied. Must be one of
    the four values enumerated below.
    """
    return {
        "title": "DatasetLoaderCode",
        "description": (
            "Per-dataset ``load.py`` source. The LLM first writes a "
            "``derivation_rationale`` framing the prediction target, "
            "then a structured ``ground_truth_provenance`` record "
            "(source mechanism, column/artifact, transform, horizon, "
            "paper quote, leakage audit), then emits ``code`` "
            "implementing those choices, then labels the result with "
            "``split_policy``."
        ),
        "type": "object",
        "properties": {
            "derivation_rationale": {"type": "string"},
            "ground_truth_provenance": {
                "type": "object",
                "description": (
                    "Auditable record of how ``ground_truth`` was "
                    "sourced. The reviewer judges fit-for-purpose "
                    "against this record; no mechanism is privileged."
                ),
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": [
                            "separate_artifact",
                            "precomputed_column",
                            "derived_transform",
                        ],
                    },
                    "column_or_artifact": {"type": "string"},
                    "transform": {"type": ["string", "null"]},
                    "horizon_units": {"type": ["string", "null"]},
                    "paper_target_quote": {"type": "string"},
                    "leakage_audit": {"type": "string"},
                },
                "required": [
                    "source",
                    "column_or_artifact",
                    "transform",
                    "horizon_units",
                    "paper_target_quote",
                    "leakage_audit",
                ],
                "additionalProperties": False,
            },
            "code": {"type": "string"},
            "split_policy": {
                "type": "string",
                "enum": [
                    "chronological_80_20",
                    "random_80_20",
                    "paper_dates",
                    "no_split",
                ],
            },
        },
        "required": [
            "derivation_rationale",
            "ground_truth_provenance",
            "code",
            "split_policy",
        ],
        "additionalProperties": False,
    }


def build_loader_review_prompt(
    *,
    loader_code: str,
    dataset_name: str,
    dataset_description: str,
    interaction_model: str,
    artifacts: list[dict],
    labeler_notes: str,
    derivation_rationale: str,
    ground_truth_provenance: dict[str, Any] | None = None,
    task_title: str = "",
    ml_task_summary: str = "",
    experiments: str = "",
    rewards: list[dict[str, Any]] | None = None,
    paper_context: str = "",
    execution_log: str = "",
    previous_script: str = "",
    previous_execution_log: str = "",
) -> str:
    """Build the loader-script reviewer prompt.

    Reuses ``review_schema()`` for output (same {analysis, approved,
    issues, suggestions} shape used for reward reviews). The reviewer
    sees the same task context the codegen LLM saw plus the loader's
    structured ``ground_truth_provenance`` record, so it can judge
    whether the chosen GT matches the paper's stated target — the
    contract is outcome-based: any sourcing mechanism (separate
    artifact / precomputed column / derived transform) is acceptable
    if it produces a GT that the rewards will faithfully score.
    """
    desc_block = (
        _truncate((dataset_description or "").strip(), field_name="dataset_description")
        or "(none)"
    )
    artifacts_block = (
        _truncate(
            json.dumps(list(artifacts or []), ensure_ascii=False, indent=2),
            field_name="artifacts",
        )
        or "[]"
    )
    labeler_notes_block = (
        _truncate((labeler_notes or "").strip(), field_name="labeler_notes")
        or "(none)"
    )
    prev_script_block = (
        _truncate(previous_script.strip(), field_name="previous_script") or "(none)"
    )
    prev_log_block = (
        _truncate(
            previous_execution_log.strip(),
            keep_tail=True,
            field_name="previous_execution_log",
        )
        or "(none)"
    )
    exec_block = (
        _truncate(execution_log.strip(), keep_tail=True, field_name="execution_log")
        or "(none)"
    )
    loader_block = (
        _truncate(loader_code.strip(), field_name="loader_code") or "(none)"
    )
    interaction_block = (interaction_model or "").strip().lower() or "(none)"
    rationale_block = (
        _truncate(
            (derivation_rationale or "").strip(),
            field_name="derivation_rationale",
        )
        or "(none)"
    )
    provenance_block = (
        _truncate(
            json.dumps(ground_truth_provenance or {}, ensure_ascii=False, indent=2),
            field_name="ground_truth_provenance",
        )
        or "{}"
    )
    task_block = _task_context_block(
        task_title=task_title,
        ml_task_summary=ml_task_summary,
        experiments=experiments,
        rewards=rewards,
        paper_context=paper_context,
    )
    return f"""
You are reviewing a per-dataset Python loader script (``load.py``). The
loader is imported by the install slicer once at install time and
called as ``load(data_dir)``; its output is demuxed into an
agent-readable HDF5 (``dataset.h5``) and a verifier-only HDF5
(``test_ground_truth.h5``).

{task_block}

Dataset:
- Name: {dataset_name}
- Description: {desc_block}
- interaction_model: {interaction_block}
- Manifest artifacts (role tags are the labeler's hint about what's
  on disk; treat as advisory):
{artifacts_block}
- Labeler notes (advisory hints, not authoritative):
{labeler_notes_block}

LLM's derivation rationale (what's being predicted, shape + split):
{rationale_block}

LLM's ground_truth_provenance (structured record of how GT was sourced):
{provenance_block}

Loader source under review:
{loader_block}

Previous round script:
{prev_script_block}

Previous round pytest log:
{prev_log_block}

Current pytest log (loader contract output):
{exec_block}

# What you are reviewing

The loader's contract is **outcome-based**: any mechanism for
producing ``ground_truth`` is acceptable (load a separate artifact,
read a precomputed column, derive a transform), as long as the
result is what the paper actually scores and the rewards can
faithfully compute against it. Do NOT reject the loader for
"not deriving" or "not using a separate file" — the choice of
mechanism is the LLM's call.

What you DO check:

* The provenance record names a real source that exists (the
  named column appears in the artifacts/preview, or the named
  artifact is in the manifest, or the named transform is what
  the code actually applies).
* The chosen target matches what the paper claims to predict —
  the ``paper_target_quote`` field should be a real, present
  passage of the task context above, and the chosen
  column/transform should plausibly produce that target.
* The chosen target is shape-compatible with the rewards listed
  in the task context (e.g. cross-sectional IC needs panel-shape
  return-like targets; FID needs distributional sample sets).
* The loader's actual emitted code is consistent with the
  provenance — e.g. ``source="precomputed_column"`` and
  ``column_or_artifact="target_return_20d"`` ⇒ the code reads
  exactly that column; ``source="derived_transform"`` ⇒ the
  code applies the named transform.
* No leakage between features and the chosen target (literal
  copies, name-substring derivatives, OHLC same-day reveals
  when the target is same-day close — see ``leakage_audit`` and
  judge whether the audit is honest).

Respond with JSON containing these fields in order:

1. analysis (string): Step-by-step. Cover:
   - Contract compliance: returns dict in B-shape (``train`` + ``test``
     each containing ``features`` and ``ground_truth``) for forecasting
     and conditional-generative, OR reference-shape (``reference``
     only) for unconditional generative. ``split_policy`` is one of
     ``chronological_80_20`` / ``random_80_20`` / ``paper_dates`` /
     ``no_split`` (the latter only with reference-shape). All ndarray
     leaves are float, finite, non-empty.
   - Per-split alignment for forecasting:
     ``len(train.features) == len(train.ground_truth)`` AND
     ``len(test.features) == len(test.ground_truth)``. B-shape
     requires BOTH ``train.ground_truth`` AND ``test.ground_truth``
     to be non-None — the install slicer omits the test side from
     the agent bundle later, but the LLM's pre-install output here
     must contain both. Reject if any required ndarray leaf is
     ``None`` or missing.
   - Provenance ↔ paper fit-for-purpose: read
     ``ground_truth_provenance.paper_target_quote`` and confirm the
     task context above actually contains that target definition. If
     the quote is invented or stretched (no nearby passage supports
     it), reject. Confirm the chosen
     ``column_or_artifact`` / ``transform`` plausibly produces that
     target.
   - Provenance ↔ code consistency: confirm the loader's emitted
     code matches the provenance record. ``source="precomputed_column"``
     should read the named column directly without re-deriving;
     ``source="derived_transform"`` should apply the stated transform;
     ``source="separate_artifact"`` should load from the named
     artifact path. If the code does something different from the
     declared source, reject.
   - Provenance ↔ rewards: are the rewards listed in the task
     context shape-compatible with the GT this loader produces?
     For example, IC / RankIC need cross-sectional panel-shape
     targets; if the loader flattens to 1D the metrics will
     degenerate. Flag (do not necessarily reject) shape mismatches —
     the dry-fire gate will catch the worst cases, but flagging
     here is cheaper.
   - Split correctness:
     * For ``chronological_80_20`` and ``paper_dates``, EVERY row in
       ``train`` MUST precede every row in ``test`` along the time
       axis (panel data: per-symbol). Sort by date BEFORE splitting.
     * For ``random_80_20``, the LLM MUST use a fixed seed.
     * For ``paper_dates``, ``split_metadata`` must record the
       explicit date boundaries used.
   - Target leakage: scan the feature column selection in the loader
     against the target named in
     ``ground_truth_provenance.column_or_artifact``. Flag (and
     reject) if any of the following are true:
     (a) a kept feature column has the target column name as a
         case-insensitive substring — e.g. target=``close`` and
         feature includes ``scaled_close``, ``pctdiff_close``,
         ``log_close``, ``norm_close``, or ``next_day_close``;
     (b) a kept feature column is a documented or named affine /
         log / pct-diff / signed-return transform of the target
         (``scaled_*``, ``norm_*``, ``z_*``, ``log_*``,
         ``pctdiff_*``, ``ret_*``, ``MACD_*``, ``RSI_*``,
         ``EMA_*``);
     (c) the target is same-day close and same-day high or low are
         used as features at the same row index, AND the target is
         NOT pre-shifted/forward-shifted;
     (d) the ``leakage_audit`` field is empty, evasive, or claims
         no drops needed when obvious drops are required.
     The pytest contract also runs a Pearson |corr| >= 0.9999
     check between every feature column and every ground_truth
     column — literal copies and affine transforms fail
     automatically. Lagged copies, OHLC same-day reveals, and
     nonlinear transforms (log, pct-diff) are NOT caught by the
     test, so this is where the reviewer must apply judgment.
   - File handling: handles non-numeric leading columns (dates,
     IDs) gracefully; uses pandas where appropriate; coerces
     dtypes; drops or fills NaNs before returning.
   - Hard-rule violations: any top-level I/O, side effects,
     hard-coded absolute paths, or heavy imports at module scope.
   - Test results: did all loader-contract pytest tests pass?
   - Previous-round fixes: if there was a prior round, were the
     issues resolved?

2. approved (boolean): true ONLY if the loader's GT is fit-for-
   purpose for THIS task, the provenance is honest and consistent
   with the code, no leakage is present, and all tests pass.

3. issues (array of strings): concise descriptions of problems
   found. Do NOT cite "wrong sourcing mechanism" alone — only cite
   problems that affect what the rewards actually score, code
   correctness, or leakage.

4. suggestions (array of strings): actionable fixes for each
   issue.
""".strip()
