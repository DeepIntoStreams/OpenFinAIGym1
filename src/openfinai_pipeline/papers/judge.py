import logging
import math
from datetime import datetime, timezone

from openfinai_pipeline.llm import LLMService
from openfinai_pipeline.papers.schemas import (
    JudgeDecision,
    JudgeLabel,
    PaperRecord,
    PaperStatus,
    ScopeDefinition,
)
from openfinai_pipeline.prompts import (
    build_paper_summary_prompt,
    build_prefilter_prompt,
    build_sift_judge_prompt,
    paper_summary_schema,
    prefilter_output_schema,
    sift_output_schema,
)
from openfinai_pipeline.settings import AppConfig, LLMConfig

logger = logging.getLogger(__name__)
PHASE_TAG = "[phase1:scrape]"


class JudgeService:
    """Multi-stage LLM judging and accepted-paper summarization."""

    def __init__(self, cfg: AppConfig, llm_cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._llm = LLMService(llm_cfg)

    def judge_scope(
        self,
        scope: ScopeDefinition,
        papers: list[PaperRecord],
        pdf_excerpts: dict[str, str] | None = None,
        sift_budget: int | None = None,
    ) -> list[JudgeDecision]:
        max_calls = (
            sift_budget
            if sift_budget is not None
            else self._cfg.judge.max_llm_calls_per_scope
        )
        decisions: list[JudgeDecision] = []
        llm_candidates: list[PaperRecord] = []
        total = len(papers)

        for index, paper in enumerate(papers, start=1):
            should_judge = True
            prefilter_score = 0.0
            prefilter_raw: dict = {}
            if self._cfg.judge.prefilter_enabled:
                should_judge, prefilter_score, prefilter_raw = self._llm_prefilter(
                    scope, paper
                )
                paper.prefilter_passed = should_judge
                paper.prefilter_score = prefilter_score
                conf = float(prefilter_raw.get("confidence_0_1", 0.0) or 0.0)
                if should_judge:
                    logger.info(
                        "%s prefilter %d/%d -> keep (score=%.1f, conf=%.2f)",
                        PHASE_TAG, index, total, prefilter_score, conf,
                    )
                else:
                    logger.info(
                        "%s prefilter %d/%d -> drop (score=%.1f, conf=%.2f)",
                        PHASE_TAG, index, total, prefilter_score, conf,
                    )
            else:
                paper.prefilter_passed = True

            if not should_judge:
                paper.status = PaperStatus.REJECTED
                decisions.append(
                    JudgeDecision(
                        paper_id=paper.paper_id,
                        scope_id=scope.id,
                        label=JudgeLabel.REJECTED,
                        score_0_10=0.0,
                        reasons="Prefilter rejected paper",
                        confidence_0_1=0.95,
                        model="prefilter_llm",
                        raw_response={
                            "prefilter_score": prefilter_score,
                            **prefilter_raw,
                        },
                    )
                )
                continue
            llm_candidates.append(paper)

        llm_queue = rank_papers_for_llm(
            llm_candidates,
            citation_boost=self._cfg.judge.ranking_citation_boost,
            recency_weight=self._cfg.judge.ranking_recency_weight,
            recency_half_life_days=self._cfg.judge.ranking_recency_half_life_days,
        )

        selected = {paper.paper_id for paper in llm_queue[:max_calls]}
        for paper in llm_candidates:
            if paper.paper_id not in selected:
                paper.status = PaperStatus.REJECTED
                decisions.append(
                    JudgeDecision(
                        paper_id=paper.paper_id,
                        scope_id=scope.id,
                        label=JudgeLabel.REJECTED,
                        score_0_10=0.0,
                        reasons="LLM call budget reached before this paper",
                        confidence_0_1=0.0,
                        model="budget_cap",
                        raw_response={},
                    )
                )

        excerpt_map = pdf_excerpts or {}
        sift_total = len(llm_queue[:max_calls])
        for sift_index, paper in enumerate(llm_queue[:max_calls], start=1):
            excerpt = excerpt_map.get(paper.paper_id)
            if not excerpt:
                decisions.append(
                    JudgeDecision(
                        paper_id=paper.paper_id,
                        scope_id=scope.id,
                        label=JudgeLabel.REJECTED,
                        score_0_10=0.0,
                        reasons="Missing fulltext excerpt for sift evidence check",
                        confidence_0_1=1.0,
                        model="sift:missing_fulltext",
                        raw_response={},
                    )
                )
                paper.status = PaperStatus.REJECTED
                logger.info(
                    "%s sift %d/%d -> reject (no excerpt)",
                    PHASE_TAG, sift_index, sift_total,
                )
                continue

            payload = self._llm.complete_or_fallback(
                build_sift_judge_prompt(scope, paper, fulltext_excerpt=excerpt),
                sift_output_schema(),
                fallback={
                    "label": "rejected",
                    "score_0_10": 0.0,
                    "confidence_0_1": 0.0,
                    "reasons": "LLM unavailable for this paper",
                    "evidence": {"experiments": "", "datasets": "", "metrics": ""},
                },
            )
            decision = JudgeDecision(
                paper_id=paper.paper_id,
                scope_id=scope.id,
                label=JudgeLabel(payload["label"]),
                score_0_10=float(payload["score_0_10"]),
                reasons=str(payload["reasons"]).strip(),
                confidence_0_1=float(payload["confidence_0_1"]),
                model=(
                    f"{payload.get('provider', 'llm')}:"
                    f"{payload.get('model', 'unknown')}"
                ),
                raw_response=payload,
            )
            decisions.append(decision)

            threshold = self._cfg.judge.threshold_default
            if (
                decision.label == JudgeLabel.ACCEPTED
                and decision.score_0_10 >= threshold
            ):
                paper.status = PaperStatus.ACCEPTED
                logger.info(
                    "%s sift %d/%d -> accept (score=%.1f, conf=%.2f)",
                    PHASE_TAG, sift_index, sift_total,
                    decision.score_0_10, decision.confidence_0_1,
                )
            else:
                paper.status = PaperStatus.REJECTED
                logger.info(
                    "%s sift %d/%d -> reject (score=%.1f, conf=%.2f)",
                    PHASE_TAG, sift_index, sift_total,
                    decision.score_0_10, decision.confidence_0_1,
                )

        return decisions

    def summarize_paper(
        self,
        scope: ScopeDefinition,
        paper: PaperRecord,
        fulltext_excerpt: str,
    ) -> dict:
        return self._llm.complete_or_fallback(
            build_paper_summary_prompt(scope, paper, fulltext_excerpt=fulltext_excerpt),
            paper_summary_schema(),
            fallback={
                "task_family": "forecasting",
                "task_name": "",
                "ml_task_summary": "",
                "experiments": "",
                "links": [],
                "datasets": [],
                "metrics": [],
            },
        )

    def enforce_downloadable_accept_cap(
        self,
        scope: ScopeDefinition,
        papers: list[PaperRecord],
        decisions: list[JudgeDecision],
        pdf_excerpts: dict[str, str],
        max_accepts: int,
    ) -> list[JudgeDecision]:
        if max_accepts < 0:
            return decisions

        paper_map = {p.paper_id: p for p in papers}
        accepted = [d for d in decisions if d.label == JudgeLabel.ACCEPTED]

        for d in accepted:
            if pdf_excerpts.get(d.paper_id):
                continue
            d.label = JudgeLabel.REJECTED
            d.reasons = _append_reason(
                d.reasons,
                "Downgraded: no valid PDF excerpt for capped final accept set.",
            )
            paper = paper_map.get(d.paper_id)
            if paper:
                paper.status = PaperStatus.REJECTED

        accepted_with_excerpt = [d for d in decisions if d.label == JudgeLabel.ACCEPTED]
        if len(accepted_with_excerpt) <= max_accepts:
            return decisions

        accepted_sorted = sorted(
            accepted_with_excerpt, key=lambda d: d.score_0_10, reverse=True
        )
        keep_ids = {d.paper_id for d in accepted_sorted[:max_accepts]}
        for d in decisions:
            if d.label != JudgeLabel.ACCEPTED or d.paper_id in keep_ids:
                continue
            d.label = JudgeLabel.REJECTED
            d.reasons = _append_reason(
                d.reasons, f"Accept cap reached for scope ({max_accepts})."
            )
            paper = paper_map.get(d.paper_id)
            if paper:
                paper.status = PaperStatus.REJECTED
        return decisions

    def _llm_prefilter(
        self, scope: ScopeDefinition, paper: PaperRecord
    ) -> tuple[bool, float, dict]:
        payload = self._llm.complete_or_fallback(
            build_prefilter_prompt(scope, paper),
            prefilter_output_schema(),
            fallback={
                "label": "rejected",
                "relevance_score_0_10": 0.0,
                "confidence_0_1": 0.0,
                "reasons": "prefilter_fallback",
            },
        )
        score = float(payload.get("relevance_score_0_10", 0.0))
        conf = float(payload.get("confidence_0_1", 0.0))
        keep = str(payload.get("label", "rejected")).strip().lower() == "accepted"
        min_score = self._cfg.judge.prefilter_llm_score_min
        min_conf = self._cfg.judge.prefilter_llm_confidence_min
        passed = keep and score >= min_score and conf >= min_conf
        return passed, score, payload


def _append_reason(existing: str, extra: str) -> str:
    existing = str(existing or "").strip()
    extra = str(extra or "").strip()
    if not existing:
        return extra
    if not extra:
        return existing
    return f"{existing} | {extra}"


def rank_papers_for_llm(
    papers: list[PaperRecord],
    *,
    citation_boost: float,
    recency_weight: float,
    recency_half_life_days: int,
) -> list[PaperRecord]:
    return sorted(
        papers,
        key=lambda p: _hybrid_priority_score(
            p,
            citation_boost=citation_boost,
            recency_weight=recency_weight,
            recency_half_life_days=recency_half_life_days,
        ),
        reverse=True,
    )


def _hybrid_priority_score(
    paper: PaperRecord,
    *,
    citation_boost: float,
    recency_weight: float,
    recency_half_life_days: int,
) -> float:
    """Multiplicative-boost ranking score.

    Prefilter relevance is the primary axis. A maximally-cited paper receives
    a ``(1 + citation_boost)x`` rank amplification but cannot overtake a
    sufficiently more relevant paper, so a weakly-relevant famous paper never
    dominates a strongly-relevant unknown one. Recency is a small additive
    sweetener on a separate axis.
    """
    prefilter = _normalize_prefilter_score(paper.prefilter_score)
    quality = _paper_quality_score(paper)
    recency = _recency_score(paper.published_at, recency_half_life_days)
    return prefilter * (1.0 + citation_boost * quality) + recency_weight * recency


def _normalize_prefilter_score(score_0_10: float | None) -> float:
    if score_0_10 is None:
        return 0.0
    return min(1.0, max(0.0, float(score_0_10) / 10.0))


def _paper_quality_score(paper: PaperRecord) -> float:
    citation = _citation_score(paper.citation_count)
    influential = _citation_score(paper.influential_citation_count, scale=200)
    venue_signal = 1.0 if (paper.journal_name or paper.venue) else 0.0
    peer_review_signal = 1.0 if paper.peer_reviewed else 0.0
    return (
        0.45 * citation
        + 0.25 * influential
        + 0.15 * venue_signal
        + 0.15 * peer_review_signal
    )


def _citation_score(value: int | None, *, scale: int = 100) -> float:
    if not value or value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(scale))


def _recency_score(published_at: str | None, half_life_days: int) -> float:
    if not published_at:
        return 0.0
    try:
        published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    now = datetime.now(tz=timezone.utc)
    age_days = max(
        0.0, (now - published.astimezone(timezone.utc)).total_seconds() / 86400.0
    )
    if half_life_days <= 0:
        return 0.0
    return 0.5 ** (age_days / float(half_life_days))
