from typing import Any

from openfinai_pipeline.prompts.corpus_builder import (
    build_real_dataset_review_prompt,
    build_synthetic_dataset_review_prompt,
    build_reward_review_prompt,
    review_schema,
)

def review_download_script(
    llm: Any | None,
    dataset_name: str,
    dataset_description: str,
    dataset_kind: str,
    script: str,
    execution_log: str,
    evidence: list[dict[str, str]] | None,
    generation_status: str,
    generation_reason: str,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    if llm is None:
        return {"approved": False, "issues": [], "suggestions": []}
    prompt = (
        build_synthetic_dataset_review_prompt(
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            script=script,
            execution_log=execution_log,
            evidence=evidence,
            generation_status=generation_status,
            generation_reason=generation_reason,
            timeout_sec=timeout_sec,
        )
        if str(dataset_kind or "").strip().lower() == "synthetic"
        else build_real_dataset_review_prompt(
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            script=script,
            execution_log=execution_log,
            evidence=evidence,
            generation_status=generation_status,
            generation_reason=generation_reason,
            timeout_sec=timeout_sec,
        )
    )
    return llm.complete_or_fallback(
        prompt,
        review_schema(),
        fallback={"approved": False, "issues": [], "suggestions": []},
    )


def review_reward_code(
    llm: Any | None,
    reward_code: str,
    reward_name: str,
    reward_description: str,
    execution_log: str,
    previous_script: str,
    previous_execution_log: str,
) -> dict[str, Any]:
    if llm is None:
        return {"approved": False, "issues": [], "suggestions": []}
    return llm.complete_or_fallback(
        build_reward_review_prompt(
            reward_code,
            reward_name=reward_name,
            reward_description=reward_description,
            execution_log=execution_log,
            previous_script=previous_script,
            previous_execution_log=previous_execution_log,
        ),
        review_schema(),
        fallback={"approved": False, "issues": [], "suggestions": []},
    )


