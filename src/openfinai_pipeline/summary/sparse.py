import math
from collections import Counter

from openfinai_pipeline.summary.chunks import SummaryChunk, tokenize


class BM25Index:
    def __init__(self, chunks: list[SummaryChunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._tokenized = [tokenize(chunk.text) for chunk in chunks]
        self._term_freqs = [Counter(tokens) for tokens in self._tokenized]
        self._doc_lengths = [len(tokens) for tokens in self._tokenized]
        self._avg_doc_length = (sum(self._doc_lengths) / len(self._doc_lengths)) if self._doc_lengths else 1.0
        self._idf = self._compute_idf()

    def scores(self, query_terms: tuple[str, ...]) -> list[float]:
        if not query_terms:
            return [0.0 for _ in self._tokenized]
        query_tokens = unique_query_tokens(query_terms)
        return [self._score_doc(doc_id, query_tokens) for doc_id in range(len(self._tokenized))]

    def _score_doc(self, doc_id: int, query_tokens: list[str]) -> float:
        tf = self._term_freqs[doc_id]
        doc_len = max(self._doc_lengths[doc_id], 1)
        score = 0.0
        for token in query_tokens:
            term_freq = tf.get(token, 0)
            if term_freq == 0:
                continue
            idf = self._idf.get(token, 0.0)
            denom = term_freq + self._k1 * (1 - self._b + self._b * doc_len / self._avg_doc_length)
            score += idf * (term_freq * (self._k1 + 1.0)) / max(denom, 1e-9)
        return score

    def _compute_idf(self) -> dict[str, float]:
        df: Counter[str] = Counter()
        for tokens in self._tokenized:
            df.update(set(tokens))
        n_docs = max(len(self._tokenized), 1)
        return {token: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()}


def unique_query_tokens(query_terms: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for term in query_terms:
        for token in tokenize(term):
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
    return tokens
