"""Structured LLM generation of per-task load.py source and rationale.

The executor owns staging, pytest validation, review, and promotion.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from openfinai_pipeline.llm import LLMService
from openfinai_pipeline.prompts.loader_builder import (
    build_write_loader_code_prompt,
    loader_code_generation_schema,
)

logger = logging.getLogger(__name__)


def _normalize_code_block(text: str) -> str:
    """Strip Markdown code fences if the LLM emitted them around the source."""
    code = text.strip()
    if code.startswith("```"):
        code = code.strip("`")
        if "\n" in code:
            code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code[:-3].rstrip()
    return code.strip()


def generate_loader_code(
    *,
    name: str,
    description: str,
    downloaded_dataset_description: str,
    interaction_model: str,
    artifacts: list[dict[str, Any]],
    preview: Any,
    labeler_notes: str,
    llm: LLMService | None,
    # Task-level context
    task_title: str = "",
    ml_task_summary: str = "",
    experiments: str = "",
    rewards: list[dict[str, Any]] | None = None,
    paper_context: str = "",
    # Round state
    execution_log: str = "",
    previous_review: str = "",
    previous_script: str = "",
) -> dict[str, Any]:
    """Run one LLM round and return the parsed structured output.

    Returns ``{derivation_rationale, ground_truth_provenance, code,
    split_policy}``. On any failure (no LLM, schema rejection,
    malformed response) returns a fallback dict with ``code=""`` so
    callers can detect the failure and record it in the round log.

    The loader contract is outcome-based: the LLM picks the
    ground_truth source mechanism (separate artifact / precomputed
    column / derived transform) itself and records the choice in
    ``ground_truth_provenance``. The reviewer judges fit-for-purpose
    against that record rather than enforcing a particular method.
    """
    fallback: dict[str, Any] = {
        "derivation_rationale": "",
        "ground_truth_provenance": {},
        "code": "",
        "split_policy": "",
    }
    if llm is None:
        return fallback
    try:
        payload = llm.complete_or_fallback(
            build_write_loader_code_prompt(
                name=name,
                description=description,
                downloaded_dataset_description=downloaded_dataset_description,
                interaction_model=interaction_model,
                artifacts=artifacts,
                preview=preview,
                labeler_notes=labeler_notes,
                task_title=task_title,
                ml_task_summary=ml_task_summary,
                experiments=experiments,
                rewards=rewards,
                paper_context=paper_context,
                previous_review=previous_review,
                previous_script=previous_script,
                execution_log=execution_log,
            ),
            loader_code_generation_schema(),
            fallback=dict(fallback),
        )
    except Exception as exc:  # noqa: BLE001 — surface LLM/router errors to caller
        logger.warning(
            "loader code generation failed dataset=%s error=%s", name, exc
        )
        return fallback
    code = _normalize_code_block(str(payload.get("code", "")))
    if not code:
        return fallback
    # Reject obviously incomplete loaders before staging; pytest remains the
    # authoritative contract validator.
    if not re.search(r"^def\s+load\s*\(", code, re.MULTILINE):
        logger.warning(
            "loader code generation: dataset=%s response missing top-level "
            "`def load(...)` — discarding",
            name,
        )
        return fallback
    has_b_shape_keys = (
        '"train"' in code or "'train'" in code
    ) and ("'test'" in code or '"test"' in code)
    has_reference_key = "'reference'" in code or '"reference"' in code
    if not (has_b_shape_keys or has_reference_key):
        logger.warning(
            "loader code generation: dataset=%s response does not appear to "
            "emit B-shape (train/test) or reference-shape (reference) keys — "
            "discarding",
            name,
        )
        return fallback
    raw_provenance = payload.get("ground_truth_provenance") or {}
    if not isinstance(raw_provenance, dict):
        raw_provenance = {}
    # Coerce keys/values to plain strings (or None for explicitly null
    # fields) so downstream callers don't need to defend against
    # provider-specific JSON-decoded types. Missing keys become empty
    # strings — the schema requires them but we tolerate older payloads
    # for backward compatibility.
    provenance: dict[str, Any] = {
        "source": str(raw_provenance.get("source") or "").strip(),
        "column_or_artifact": str(
            raw_provenance.get("column_or_artifact") or ""
        ).strip(),
        "transform": (
            str(raw_provenance["transform"]).strip()
            if raw_provenance.get("transform") is not None
            else None
        ),
        "horizon_units": (
            str(raw_provenance["horizon_units"]).strip()
            if raw_provenance.get("horizon_units") is not None
            else None
        ),
        "paper_target_quote": str(
            raw_provenance.get("paper_target_quote") or ""
        ).strip(),
        "leakage_audit": str(
            raw_provenance.get("leakage_audit") or ""
        ).strip(),
    }
    return {
        "derivation_rationale": str(
            payload.get("derivation_rationale") or ""
        ).strip(),
        "ground_truth_provenance": provenance,
        "code": code,
        "split_policy": str(payload.get("split_policy") or "").strip(),
    }
