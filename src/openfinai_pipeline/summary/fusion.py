
def reciprocal_rank_fusion(rankings: list[list[int]], *, k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if abs(high - low) < 1e-12:
        return [0.0 if high == 0.0 else 1.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]
