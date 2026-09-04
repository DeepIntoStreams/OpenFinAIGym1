import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from openfinai_pipeline.papers.schemas import PaperRecord, SourceName
from openfinai_pipeline.settings import SemanticScholarConfig

logger = logging.getLogger(__name__)

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_MATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
S2_FIELDS = (
    "citationCount,influentialCitationCount,paperId,venue,journal,publicationTypes"
)
_REQUEST_DELAY_SEC = 0.02
_TITLE_MATCH_DELAY_SEC = 1.1
_CACHE_TTL_DAYS = 7
_TITLE_MATCH_MIN_LEN = 12


class SemanticScholarClient:
    """Semantic Scholar enrichment with local JSON cache."""

    def __init__(self, cfg: SemanticScholarConfig) -> None:
        self._cfg = cfg
        self._session = requests.Session()
        self._headers = _get_headers()
        self._cache_path = Path(cfg.cache_path)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()

    def enrich(self, papers: list[PaperRecord]) -> list[PaperRecord]:
        """Batch-enrich papers and reuse cached results when valid."""
        if not self._cfg.enabled or not papers:
            return papers

        batch_size = max(1, min(self._cfg.batch_size, 500))
        dirty = False

        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            unresolved: list[PaperRecord] = []
            for paper in batch:
                cached = self._read_cache(paper.paper_id)
                if cached:
                    cached_citation = _to_int(cached.get("citation_count"))
                    if cached_citation is not None:
                        paper.citation_count = cached_citation
                    cached_influential = _to_int(
                        cached.get("influential_citation_count")
                    )
                    if cached_influential is not None:
                        paper.influential_citation_count = cached_influential
                    cached_sid = _to_str(cached.get("semantic_scholar_id"))
                    if cached_sid:
                        paper.semantic_scholar_id = cached_sid
                    if not paper.venue:
                        paper.venue = _to_str(cached.get("venue"))
                    if not paper.journal_name:
                        paper.journal_name = _to_str(cached.get("journal_name"))
                    cached_types = _to_string_list(cached.get("publication_types"))
                    if not paper.publication_types:
                        paper.publication_types = cached_types
                    elif cached_types:
                        paper.publication_types = _merge_types(
                            paper.publication_types, cached_types
                        )
                    paper.peer_reviewed = paper.peer_reviewed or bool(
                        cached.get("peer_reviewed")
                    )
                else:
                    unresolved.append(paper)

            if unresolved:
                id_mappings: list[tuple[PaperRecord, str]] = []
                title_fallbacks: list[PaperRecord] = []
                for paper in unresolved:
                    sid = _semantic_scholar_id(paper)
                    if sid:
                        id_mappings.append((paper, sid))
                    elif paper.title.strip():
                        title_fallbacks.append(paper)

                ids = [sid for _, sid in id_mappings]
                results = self._fetch_batch(ids) if ids else []
                for (paper, _), result in zip(id_mappings, results):
                    if result and self._apply_s2_result(paper, result):
                        dirty = True

                for paper in title_fallbacks:
                    result = self._title_match(paper)
                    if result and self._apply_s2_result(paper, result):
                        dirty = True
                    time.sleep(_TITLE_MATCH_DELAY_SEC)

            time.sleep(_REQUEST_DELAY_SEC)

        if dirty:
            self._save_cache()
        return papers

    def _apply_s2_result(self, paper: PaperRecord, result: dict) -> bool:
        """Merge one S2 result dict onto ``paper`` and update cache. Returns True on change."""
        s2_citation = _to_int(result.get("citationCount"))
        s2_influential = _to_int(result.get("influentialCitationCount"))
        s2_sid = _to_str(result.get("paperId"))
        s2_venue = _to_str(result.get("venue"))
        s2_journal = _extract_journal_name(result.get("journal"))
        s2_types = _to_string_list(result.get("publicationTypes"))

        if s2_citation is not None:
            paper.citation_count = s2_citation
        if s2_influential is not None:
            paper.influential_citation_count = s2_influential
        if s2_sid:
            paper.semantic_scholar_id = s2_sid
        if not paper.venue and s2_venue:
            paper.venue = s2_venue
        if not paper.journal_name and s2_journal:
            paper.journal_name = s2_journal
        if not paper.publication_types:
            paper.publication_types = s2_types
        elif s2_types:
            paper.publication_types = _merge_types(paper.publication_types, s2_types)
        paper.peer_reviewed = (
            _infer_peer_reviewed(paper.publication_types, paper.journal_name)
            or paper.peer_reviewed
        )
        self._write_cache(
            paper.paper_id,
            citation_count=paper.citation_count,
            semantic_scholar_id=paper.semantic_scholar_id,
            influential_citation_count=paper.influential_citation_count,
            venue=paper.venue,
            journal_name=paper.journal_name,
            publication_types=paper.publication_types,
            peer_reviewed=paper.peer_reviewed,
        )
        return True

    def _title_match(self, paper: PaperRecord) -> dict | None:
        """Look up a paper by title via S2 ``/paper/search/match``.

        Used as a final fallback for papers with no S2-usable ID (e.g.
        manually-imported PDFs without a DOI). S2 returns the single best
        title match; we apply it as-is.
        """
        title = (paper.title or "").strip()
        if len(title) < _TITLE_MATCH_MIN_LEN:
            return None
        try:
            resp = self._session.get(
                S2_MATCH_URL,
                params={"query": title, "fields": S2_FIELDS},
                headers=self._headers,
                timeout=self._cfg.timeout_sec,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            payload = resp.json()
        except requests.HTTPError as e:
            response = e.response
            status_code = response.status_code if response is not None else "unknown"
            logger.debug(
                "semantic scholar title match failed status=%s title=%s",
                status_code,
                title[:120],
            )
            return None
        except requests.RequestException as e:
            logger.debug(
                "semantic scholar title match failed title=%s err=%s",
                title[:120],
                e,
            )
            return None
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or not data:
            return None
        first = data[0]
        return first if isinstance(first, dict) else None

    def _fetch_batch(self, ids: list[str]) -> list[dict | None]:
        """Fetch one S2 batch and return row-aligned results.

        On HTTP 400 ("No valid paper ids given") S2 returns an error for the
        *entire* batch when none of the ids resolve. We recursively split the
        batch in half to isolate the poison ids so valid ones are still
        enriched. Batches of size 1 that 400 are accepted as unresolved.
        """
        if not ids:
            return []
        try:
            resp = self._session.post(
                S2_BATCH_URL,
                json={"ids": ids},
                params={"fields": S2_FIELDS},
                headers=self._headers,
                timeout=self._cfg.timeout_sec,
            )
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, list):
                return [None] * len(ids)
            if len(payload) < len(ids):
                payload.extend([None] * (len(ids) - len(payload)))
            return payload[: len(ids)]
        except requests.HTTPError as e:
            response = e.response
            status_code = response.status_code if response is not None else None
            if status_code == 400 and len(ids) > 1:
                mid = len(ids) // 2
                logger.info(
                    "semantic scholar batch 400; splitting size=%d into %d+%d",
                    len(ids),
                    mid,
                    len(ids) - mid,
                )
                time.sleep(_REQUEST_DELAY_SEC)
                left = self._fetch_batch(ids[:mid])
                time.sleep(_REQUEST_DELAY_SEC)
                right = self._fetch_batch(ids[mid:])
                return left + right
            body = ""
            if response is not None:
                try:
                    body = response.text
                except Exception:
                    body = ""
            logger.warning(
                "semantic scholar request failed status=%s ids=%d sample_ids=%s body=%s",
                status_code if status_code is not None else "unknown",
                len(ids),
                ids[:5],
                (body or "").strip().replace("\n", " ")[:500],
            )
            return [None] * len(ids)
        except requests.RequestException as e:
            logger.warning(
                "semantic scholar request failed ids=%d sample_ids=%s err=%s",
                len(ids),
                ids[:5],
                e,
            )
            return [None] * len(ids)

    def _load_cache(self) -> dict[str, dict]:
        if not self._cache_path.exists():
            return {}
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if isinstance(raw, dict):
            return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
        return {}

    def _save_cache(self) -> None:
        self._cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _read_cache(self, paper_id: str) -> dict | None:
        """Return cached S2 metadata if entry is within TTL."""
        ttl = datetime.now(tz=timezone.utc) - timedelta(days=_CACHE_TTL_DAYS)
        row = self._cache.get(paper_id)
        if not isinstance(row, dict):
            return None
        cached_at_raw = row.get("cached_at")
        if not isinstance(cached_at_raw, str):
            return None
        try:
            cached_at = datetime.fromisoformat(cached_at_raw)
        except ValueError:
            return None
        if cached_at < ttl:
            return None
        return {
            "citation_count": row.get("citation_count"),
            "semantic_scholar_id": row.get("semantic_scholar_id"),
            "influential_citation_count": row.get("influential_citation_count"),
            "venue": row.get("venue"),
            "journal_name": row.get("journal_name"),
            "publication_types": row.get("publication_types", []),
            "peer_reviewed": bool(row.get("peer_reviewed")),
        }

    def _write_cache(
        self,
        paper_id: str,
        *,
        citation_count: int | None,
        semantic_scholar_id: str | None,
        influential_citation_count: int | None,
        venue: str | None,
        journal_name: str | None,
        publication_types: list[str],
        peer_reviewed: bool,
    ) -> None:
        """Upsert one cache entry after successful S2 enrichment."""
        self._cache[paper_id] = {
            "citation_count": citation_count,
            "semantic_scholar_id": semantic_scholar_id,
            "influential_citation_count": influential_citation_count,
            "venue": venue,
            "journal_name": journal_name,
            "publication_types": publication_types,
            "peer_reviewed": peer_reviewed,
            "cached_at": datetime.now(tz=timezone.utc).isoformat(),
        }


def _get_headers() -> dict[str, str]:
    """Build optional API key header for Semantic Scholar."""
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    return {"x-api-key": key} if key else {}


def _semantic_scholar_id(paper: PaperRecord) -> str | None:
    """Map local paper record to S2-supported ID format.

    Prefer ``ARXIV:<id>`` for arxiv-sourced papers because S2 always indexes
    arXiv IDs, while many author-registered DOIs (SSRN ``10.2139/ssrn.*``,
    Research Square ``10.21203/rs.*``, small regional journals) are not in S2
    and cause the whole batch to fail with HTTP 400. Non-arxiv papers without
    a DOI return ``None`` and fall through to the title-match path.
    """
    if paper.source == SourceName.ARXIV and paper.paper_id:
        return f"ARXIV:{paper.paper_id}"
    if paper.doi:
        return f"DOI:{paper.doi}"
    return None


def _to_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _to_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _extract_journal_name(value: object) -> str | None:
    if isinstance(value, str):
        return _to_str(value)
    if isinstance(value, dict):
        return _to_str(value.get("name"))
    return None


def _to_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                out.append(cleaned)
    return out


def _merge_types(current: list[str], incoming: list[str]) -> list[str]:
    merged = list(current)
    seen = {item.lower() for item in current}
    for item in incoming:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _infer_peer_reviewed(
    publication_types: list[str], journal_name: str | None
) -> bool:
    if journal_name:
        return True
    normalized = {t.lower().replace(" ", "") for t in publication_types if t}
    return any(tag in normalized for tag in {"journalarticle", "reviewarticle"})
