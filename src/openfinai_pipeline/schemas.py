from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class JudgeLabel(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class PaperStatus(str, Enum):
    SCRAPED = "scraped"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class SourceName(str, Enum):
    ARXIV = "arxiv"
    MANUAL = "manual"


class ScopeDefinition(BaseModel):
    id: str
    name: str
    description: str
    enabled: bool = True
    queries: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


class PaperRecord(BaseModel):
    paper_id: str
    source: SourceName = SourceName.ARXIV
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published_at: str | None = None
    updated_at: str | None = None
    pdf_url: str | None = None
    primary_category: str | None = None
    doi: str | None = None
    raw_payload_hash: str
    scope_ids: list[str] = Field(default_factory=list)
    citation_count: int | None = None
    semantic_scholar_id: str | None = None
    influential_citation_count: int | None = None
    venue: str | None = None
    journal_name: str | None = None
    publication_types: list[str] = Field(default_factory=list)
    peer_reviewed: bool = False
    prefilter_score: float = 0.0
    prefilter_passed: bool = False
    status: PaperStatus = PaperStatus.SCRAPED


class JudgeDecision(BaseModel):
    paper_id: str
    scope_id: str
    label: JudgeLabel
    score_0_10: float = Field(ge=0.0, le=10.0)
    reasons: str = ""
    confidence_0_1: float = Field(ge=0.0, le=1.0)
    model: str
    raw_response: dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    run_id: str
    started_at: str
    finished_at: str | None = None
    scope_ids: list[str] = Field(default_factory=list)
    total_scraped: int = 0
    total_enriched: int = 0
    total_prefiltered: int = 0
    total_judged: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    total_downloaded: int = 0
    total_tasks_constructed: int = 0
    total_tasks_failed: int = 0
    # Phase 1 rerun-dedupe counters: papers dropped post-collection because
    # this scope's index.json already records a prior judgment for them.
    total_skipped_seen_accepted: int = 0
    total_skipped_seen_rejected: int = 0
    errors: list[str] = Field(default_factory=list)
    status: str = "running"


class TaskIndexEntry(BaseModel):
    paper_id_num: int
    paper_dir: str
    run_id: str
    scope_id: str
    paper_id: str
    title: str
    label: str = "accepted"
    updated_at: str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


class SeenRejectedEntry(BaseModel):
    """Per-scope log of papers previously judged-rejected.

    Lets Phase 1 scrape skip re-judging the same paper on rerun without
    burning prefilter/sift LLM credits and bandwidth. The same paper_id can
    still be accepted under a *different* scope — this log is per-scope,
    mirroring `index.json`'s scope partitioning.

    Disjoint from `TaskIndex.entries`: a paper is in `entries` (accepted)
    OR `seen_rejected` (rejected), never both. `_prepare_target` enforces
    this by dropping any matching `seen_rejected` entry on acceptance, and
    `record_rejected` skips paper_ids already present in `entries`.
    """

    paper_id: str
    last_run_id: str
    last_judged_at: str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    last_reason: str = ""
    last_model: str = ""


class TaskIndex(BaseModel):
    next_paper_id: int = 1
    entries: list[TaskIndexEntry] = Field(default_factory=list)
    seen_rejected: list[SeenRejectedEntry] = Field(default_factory=list)


# Phase 1c — Trading triage records


class TradingTriageStatus(str, Enum):
    ROUTED = "routed"
    PARTIAL_MATCH = "partial_match"
    NO_MATCH = "no_match"
    NOVEL_TASK_REQUIRED = "novel_task_required"


class ConfigProvenance(BaseModel):
    source: Literal["explicit", "inferred", "catalog_default"]
    evidence: str | None = None


class TradingTriageAlternate(BaseModel):
    routed_from: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class TradingTriageRecord(BaseModel):
    """Phase 1c output: per-paper binding to a curated trading task."""

    schema_version: Literal["1"] = "1"
    catalog_version: str
    catalog_hash: str
    paper_id: str
    model: str
    status: TradingTriageStatus
    routed_from: str | None = None
    recommended_task_id: str | None = None
    recommended_task_class: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    task_config: dict[str, Any] = Field(default_factory=dict)
    config_provenance: dict[str, ConfigProvenance] = Field(default_factory=dict)
    alternates: list[TradingTriageAlternate] = Field(default_factory=list)
    gap_description: str | None = None
    novel_features: list[str] = Field(default_factory=list)


class PaperTaskBinding(BaseModel):
    """Schema for paper.json["task"] under the unified `kind` discriminator.

    Backward compat: readers should treat an absent `kind` as "auto_generated".
    """

    kind: Literal["auto_generated", "curated_routed"]
    task_dir: str
    task_id: str | None = None
    generated: bool | None = None
    routed_from: str | None = None
    triage_record_path: str | None = None


def triage_output_schema() -> dict[str, Any]:
    """JSON schema for the subset of TradingTriageRecord the LLM produces.

    The orchestrator stamps catalog_version, catalog_hash, paper_id, model,
    recommended_task_id, recommended_task_class post-hoc.
    """
    # Property ORDER matters: structured-output LLMs emit in declared order, so
    # reasoning/extraction fields come FIRST and committed decisions LAST. This
    # uses sequential generation as a scratchpad — by the time the model picks a
    # `status` and `routed_from`, it has already written out the rationale and
    # the per-field `task_config`/`config_provenance`, so the decision is grounded
    # in concrete extraction rather than a snap judgement filled in with defaults.
    return {
        "title": "TradingTriageDecision",
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "confidence", "rationale"],
        "properties": {
            "rationale": {
                "type": "string",
                "description": (
                    "SCRATCHPAD — emit this FIRST and reason carefully here before "
                    "filling in any other field. Cover: paper's asset class and "
                    "exact instruments, backtest timeframe (start/end + bar "
                    "resolution), action interface (long/short/flat, sizing, "
                    "order types), headline evaluation metric(s), and how each of "
                    "those maps (or fails to map) to one of the curated tasks in "
                    "the catalog. This grounds every subsequent field."
                ),
            },
            "task_config": {
                "type": "object",
                "description": (
                    "Per-paper config for the chosen curated task. When status "
                    "is `routed` or `partial_match`, you MUST populate the "
                    "values you discussed in `rationale` — at minimum the "
                    "fields: `symbols`, `target_symbols`, `start`, `end`, "
                    "`data_resolution`, `episode_length`. Each populated field "
                    "MUST have a matching `config_provenance` entry with "
                    "`source='explicit'` (quote the paper) or `source='inferred'` "
                    "(explain). The catalog defaults are NOT a preferred answer; "
                    "they fill in ONLY for fields the paper genuinely never "
                    "addresses (typically `initial_cash`, `slippage_pct`, "
                    "`transaction_cost_pct` when the paper does not state them)."
                ),
            },
            "config_provenance": {
                "type": "object",
                "description": (
                    "Per-field provenance for every key you placed in "
                    "`task_config`: 'explicit' (paper directly states the value, "
                    "include a quote in `evidence`) or 'inferred' (paper implies "
                    "it; explain). Do NOT emit 'catalog_default' entries — those "
                    "are stamped by the orchestrator for fields you omit."
                ),
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["source"],
                    "properties": {
                        "source": {
                            "type": "string",
                            "enum": ["explicit", "inferred", "catalog_default"],
                        },
                        "evidence": {
                            "type": ["string", "null"],
                            "description": "Quote from the paper or null for catalog_default.",
                        },
                    },
                },
            },
            "gap_description": {
                "type": ["string", "null"],
                "description": (
                    "After the extraction above, describe what the paper needs "
                    "that the curated task cannot represent. Null if the fit is "
                    "clean. Drives the status decision below."
                ),
            },
            "novel_features": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Structural features the catalog cannot represent (e.g. "
                    "'options spreads', 'continuous position sizing')."
                ),
            },
            "routed_from": {
                "type": ["string", "null"],
                "description": (
                    "Base curated task id (e.g. 'offline_crypto_trading'), picked "
                    "now that the extraction and gaps above are known. Required "
                    "when status is 'routed' or 'partial_match'; null otherwise."
                ),
            },
            "status": {
                "type": "string",
                "enum": [s.value for s in TradingTriageStatus],
                "description": (
                    "Final decision, informed by everything above. routed: clean "
                    "fit; partial_match: best fit but gaps exist; no_match: no "
                    "curated task is appropriate; novel_task_required: paper "
                    "proposes a task structure the catalog cannot represent."
                ),
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "alternates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["routed_from", "confidence", "reason"],
                    "properties": {
                        "routed_from": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    }


def triage_fallback_payload() -> dict[str, Any]:
    """Default LLM output when the call fails — treats the paper as no_match."""
    return {
        "status": TradingTriageStatus.NO_MATCH.value,
        "routed_from": None,
        "confidence": 0.0,
        "rationale": "LLM call failed; falling back to no_match.",
        "task_config": {},
        "config_provenance": {},
        "alternates": [],
        "gap_description": "Triage LLM call failed; no routing attempted.",
        "novel_features": [],
    }
