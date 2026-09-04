import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = _load_cross_encoder(model_name)

    @property
    def available(self) -> bool:
        return self._model is not None

    def scores(self, query_text: str, chunk_texts: list[str]) -> list[float]:
        if not self._model or not chunk_texts:
            return [0.0 for _ in chunk_texts]
        pairs = [[query_text, chunk_text] for chunk_text in chunk_texts]
        try:
            raw_scores = self._model.predict(pairs, show_progress_bar=False)  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover
            logger.warning("summary semantic rerank failed, falling back to prereank-only: %s", exc)
            return [0.0 for _ in chunk_texts]
        return [float(score) for score in raw_scores]


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> Any | None:
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
    except ImportError:
        logger.info("sentence-transformers cross-encoder unavailable; summary retrieval will use prereank-only selection")
        return None
    try:
        return CrossEncoder(model_name)
    except Exception as exc:  # pragma: no cover
        logger.warning("failed to load summary reranker model %s: %s", model_name, exc)
        return None
