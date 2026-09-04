import logging
from functools import lru_cache
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = _load_embedding_model(model_name)

    @property
    def available(self) -> bool:
        return self._model is not None

    def embed_documents(self, chunk_texts: list[str]) -> np.ndarray | None:
        if not self._model or not chunk_texts:
            return None
        embeddings = self._encode(chunk_texts)
        if embeddings is None or len(embeddings) != len(chunk_texts):
            return None
        return embeddings

    def scores(self, query_text: str, doc_embeddings: np.ndarray | None) -> list[float]:
        if not self._model or doc_embeddings is None or len(doc_embeddings) == 0:
            return [0.0 for _ in range(0 if doc_embeddings is None else len(doc_embeddings))]
        query_embedding = self._encode([query_text])
        if query_embedding is None or len(query_embedding) != 1:
            return [0.0 for _ in range(len(doc_embeddings))]
        query_vec = query_embedding[0]
        doc_vecs = doc_embeddings
        denom = (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)).clip(min=1e-9)
        sims = np.dot(doc_vecs, query_vec) / denom
        return [float(score) for score in sims.tolist()]

    def _encode(self, texts: list[str]) -> np.ndarray | None:
        try:
            vectors = self._model.encode(texts, normalize_embeddings=False, show_progress_bar=False)  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover
            logger.warning("summary dense retrieval encode failed, falling back to sparse-only: %s", exc)
            return None
        return np.asarray(vectors, dtype=float)


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str) -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.info("sentence-transformers not installed; summary retrieval will use sparse-only ranking")
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover
        logger.warning("failed to load summary embedding model %s: %s", model_name, exc)
        return None
