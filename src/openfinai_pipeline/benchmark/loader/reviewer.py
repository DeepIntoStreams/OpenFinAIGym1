"""LLM reviewer for per-task ``load.py`` candidates.

Mirrors the round-loop shape of the reward reviewer at
``corpus/reviewer.py`` so ``benchmark/loader/executor.py`` can drive
identical generation/validation/review iterations. Sees the same
task-level context as the codegen prompt (paper, summary, experiments,
rewards) so it can flag a target choice that doesn't match what the
task actually predicts.
"""

from __future__ import annotations

from typing import Any

from openfinai_pipeline.prompts.loader_builder import (
    build_loader_review_prompt,
    review_schema,
)


def review_loader_code(
    llm: Any | None,
    *,
    loader_code: str,
    dataset_name: str,
    dataset_description: str,
    interaction_model: str,
    artifacts: list[dict[str, Any]],
    labeler_notes: str,
    derivation_rationale: str,
    ground_truth_provenance: dict[str, Any] | None = None,
    # Task-level context
    task_title: str = "",
    ml_task_summary: str = "",
    experiments: str = "",
    rewards: list[dict[str, Any]] | None = None,
    paper_context: str = "",
    # Round state
    execution_log: str = "",
    previous_script: str = "",
    previous_execution_log: str = "",
) -> dict[str, Any]:
    """Reviewer for the per-task ``load.py`` loader script.

    Reuses ``review_schema()`` so the round-loop logic in
    ``benchmark/loader/executor.py`` matches the reward reviewer's
    ``_rewrite_reward_rounds`` shape exactly. The reviewer judges
    the loader's GT against the structured ``ground_truth_provenance``
    record produced by the codegen LLM — no sourcing mechanism is
    privileged.
    """
    if llm is None:
        return {"approved": False, "issues": [], "suggestions": []}
    return llm.complete_or_fallback(
        build_loader_review_prompt(
            loader_code=loader_code,
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            interaction_model=interaction_model,
            artifacts=artifacts,
            labeler_notes=labeler_notes,
            derivation_rationale=derivation_rationale,
            ground_truth_provenance=ground_truth_provenance,
            task_title=task_title,
            ml_task_summary=ml_task_summary,
            experiments=experiments,
            rewards=rewards,
            paper_context=paper_context,
            execution_log=execution_log,
            previous_script=previous_script,
            previous_execution_log=previous_execution_log,
        ),
        review_schema(),
        fallback={"approved": False, "issues": [], "suggestions": []},
    )
