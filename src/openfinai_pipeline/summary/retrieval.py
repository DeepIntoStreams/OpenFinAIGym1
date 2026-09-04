import logging
import re

import numpy as np

from openfinai_pipeline.settings import SummaryConfig
from openfinai_pipeline.summary.chunks import SummaryChunk
from openfinai_pipeline.summary.dense import DenseRetriever
from openfinai_pipeline.summary.fusion import normalize_scores, reciprocal_rank_fusion
from openfinai_pipeline.summary.intents import RetrievalIntent, retrieval_intents
from openfinai_pipeline.summary.reranker import CrossEncoderReranker
from openfinai_pipeline.summary.sparse import BM25Index
from openfinai_pipeline.utils.logging import log_detail

logger = logging.getLogger(__name__)
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)


def rank_summary_chunks(
    chunks: list[SummaryChunk],
    *,
    config: SummaryConfig,
) -> dict[int, dict[str, float]]:
    index = BM25Index(chunks)
    dense = DenseRetriever(config.embedding_model)
    reranker = CrossEncoderReranker(config.reranker_model)
    ranked: dict[int, dict[str, float]] = {}
    chunk_texts = [chunk.text for chunk in chunks]
    doc_embeddings = dense.embed_documents(chunk_texts)
    log_detail(
        logger,
        "summary retrieval start chunks=%d intents=%d",
        len(chunks),
        len(retrieval_intents()),
    )
    log_detail(
        logger,
        "summary retrieval models dense_available=%s dense_model=%s reranker_available=%s reranker_model=%s doc_embeddings_cached=%s candidate_top_k=%d",
        dense.available,
        getattr(dense, "_model_name", "n/a"),
        reranker.available,
        getattr(reranker, "_model_name", "n/a"),
        doc_embeddings is not None,
        config.candidate_top_k,
    )
    for chunk_id, chunk in enumerate(chunks):
        ranked[chunk_id] = {"front_matter": _score_front_matter(chunk)}

    for intent in retrieval_intents():
        intent_scores = _rank_intent(
            index=index,
            dense=dense,
            reranker=reranker,
            chunks=chunks,
            doc_embeddings=doc_embeddings,
            intent=intent,
            config=config,
        )
        for chunk_id, score in intent_scores.items():
            ranked.setdefault(chunk_id, {})
            ranked[chunk_id][intent.name] = score

    for chunk_id in range(len(chunks)):
        for intent in retrieval_intents():
            ranked[chunk_id].setdefault(intent.name, 0.0)
        ranked[chunk_id]["overall"] = sum(ranked[chunk_id].values())
    log_detail(
        logger,
        "summary retrieval complete chunks=%d top_overall=%s",
        len(chunks),
        _format_top_chunks(ranked, chunks, "overall", top_k=6),
    )
    return ranked


def _rank_intent(
    *,
    index: BM25Index,
    dense: DenseRetriever,
    reranker: CrossEncoderReranker,
    chunks: list[SummaryChunk],
    doc_embeddings: np.ndarray | None,
    intent: RetrievalIntent,
    config: SummaryConfig,
) -> dict[int, float]:
    bm25_scores = index.scores(intent.terms)
    dense_scores = dense.scores(
        intent.query_text or " ".join(intent.terms),
        doc_embeddings if hasattr(doc_embeddings, "__len__") else None,
    )
    if len(dense_scores) != len(chunks):
        dense_scores = [0.0 for _ in chunks]
    bm25_rank = _rank_ids_from_scores(bm25_scores)
    rankings = [bm25_rank]
    if dense.available:
        dense_rank = _rank_ids_from_scores(dense_scores)
        rankings.append(dense_rank)
    fused = reciprocal_rank_fusion(rankings)
    fused_scores = [fused.get(chunk_id, 0.0) for chunk_id in range(len(chunks))]
    normalized_fused = normalize_scores(fused_scores)
    rerank_candidates = sorted(fused, key=lambda chunk_id: fused[chunk_id], reverse=True)[: config.candidate_top_k]

    final_scores, semantic_top_ids = _build_final_scores(
        chunks=chunks,
        reranker=reranker,
        candidate_ids=rerank_candidates,
        normalized_fused=normalized_fused,
        query_text=intent.query_text or " ".join(intent.terms),
        intent=intent,
        config=config,
    )
    log_detail(
        logger,
        "summary intent=%s bm25_top=%s dense_top=%s fused_top=%s rerank_candidate_count=%d semantic_top=%s final_top=%s",
        intent.name,
        _format_ranked_ids(_rank_ids_from_scores(bm25_scores), chunks),
        _format_ranked_ids(_rank_ids_from_scores(dense_scores), chunks) if dense.available else "sparse-only",
        _format_ranked_ids(rerank_candidates, chunks),
        len(rerank_candidates),
        _format_ranked_ids(semantic_top_ids, chunks),
        _format_ranked_ids(sorted(final_scores, key=final_scores.get, reverse=True)[:3], chunks),
    )
    return final_scores


def _build_final_scores(
    *,
    chunks: list[SummaryChunk],
    reranker: CrossEncoderReranker,
    candidate_ids: list[int],
    normalized_fused: list[float],
    query_text: str,
    intent: RetrievalIntent,
    config: SummaryConfig,
) -> tuple[dict[int, float], list[int]]:
    final_scores = {
        chunk_id: 0.001 * normalized_fused[chunk_id]
        for chunk_id in range(len(chunks))
    }
    if not candidate_ids:
        return final_scores, []

    if not reranker.available:
        for chunk_id in candidate_ids:
            final_scores[chunk_id] = 0.05 + 0.95 * normalized_fused[chunk_id]
        return final_scores, candidate_ids[:3]

    semantic_scores = reranker.scores(query_text, [chunks[chunk_id].text for chunk_id in candidate_ids])
    normalized_semantic = normalize_scores(semantic_scores)
    heuristic_scores = normalize_scores([_heuristic_prior(chunks[chunk_id], intent) for chunk_id in candidate_ids])
    semantic_weight, heuristic_weight = _normalized_weights(config)
    for index, chunk_id in enumerate(candidate_ids):
        final_scores[chunk_id] = (
            semantic_weight * normalized_semantic[index]
            + heuristic_weight * heuristic_scores[index]
        )
    return final_scores, _semantic_top_ids(candidate_ids, semantic_scores)


def _semantic_top_ids(
    candidate_ids: list[int],
    semantic_scores: list[float],
) -> list[int]:
    ranked = sorted(
        zip(candidate_ids, semantic_scores, strict=False),
        key=lambda item: item[1],
        reverse=True,
    )
    return [chunk_id for chunk_id, _ in ranked[:3]]




def _score_front_matter(chunk: SummaryChunk) -> float:
    lowered_section = chunk.section_title.lower()
    lowered_text = chunk.text.lower()
    score = 0.0
    if chunk.page_number <= 2:
        score += 10.0
    if "abstract" in lowered_section:
        score += 16.0
    if "introduction" in lowered_section:
        score += 8.0
    if "we propose" in lowered_text or "this paper" in lowered_text:
        score += 3.0
    return score


def _heuristic_prior(chunk: SummaryChunk, intent: RetrievalIntent) -> float:
    lowered_text = chunk.text.lower()
    lowered_section = chunk.section_title.lower()
    bonus = 0.0
    for phrase in intent.terms:
        if " " in phrase and phrase in lowered_text:
            bonus += 1.0
    for section_term in intent.section_terms:
        if section_term in lowered_section:
            bonus += 2.0
    if _URL_PATTERN.search(chunk.text):
        bonus += 1.0
    if ("appendix" in lowered_section or "supplementary" in lowered_section) and bonus > 0:
        bonus += 0.75
    if _looks_like_reference_text(lowered_text):
        bonus -= 2.0
    return bonus


def _normalized_weights(config: SummaryConfig) -> tuple[float, float]:
    total = config.semantic_weight + config.heuristic_weight
    if total <= 0:
        return 1.0, 0.0
    return config.semantic_weight / total, config.heuristic_weight / total


def _rank_ids_from_scores(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda chunk_id: scores[chunk_id], reverse=True)


def _format_ranked_ids(chunk_ids: list[int], chunks: list[SummaryChunk], top_k: int = 3) -> str:
    parts: list[str] = []
    for chunk_id in chunk_ids[:top_k]:
        chunk = chunks[chunk_id]
        section = chunk.section_title or "no-section"
        parts.append(f"{chunk_id}:p{chunk.page_number}:{section[:24]}")
    return ", ".join(parts) or "(none)"


def _format_top_chunks(
    ranked: dict[int, dict[str, float]],
    chunks: list[SummaryChunk],
    key: str,
    *,
    top_k: int,
) -> str:
    chunk_ids = sorted(ranked, key=lambda chunk_id: ranked[chunk_id].get(key, 0.0), reverse=True)
    return _format_ranked_ids(chunk_ids, chunks, top_k=top_k)


def _looks_like_reference_text(lowered_text: str) -> bool:
    head = lowered_text[:300]
    return (
        head.startswith("references")
        or head.startswith("bibliography")
        or ("references" in head and "[" in head)
        or ("bibliography" in head and "[" in head)
    )
