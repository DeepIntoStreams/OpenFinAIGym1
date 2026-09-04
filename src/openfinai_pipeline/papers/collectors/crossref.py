import logging
import os
import re
import time
from difflib import SequenceMatcher

import requests

from openfinai_pipeline.papers.schemas import PaperRecord
from openfinai_pipeline.settings import CrossrefConfig

logger = logging.getLogger(__name__)

CROSSREF_WORKS_URL = "https://api.crossref.org/works"
_SPACE = re.compile(r"\s+")
_REQUEST_DELAY_SEC = 0.1


class CrossrefClient:
    """Crossref enrichment for DOI and lightweight venue metadata."""

    def __init__(self, cfg: CrossrefConfig) -> None:
        self._cfg = cfg
        self._session = requests.Session()
        self._mailto = cfg.mailto or os.getenv("CROSSREF_MAILTO")

    def enrich_doi(self, papers: list[PaperRecord]) -> list[PaperRecord]:
        """Fill missing DOI and backfill basic venue/quality metadata."""
        if not self._cfg.enabled:
            return papers
        for paper in papers:
            if not _needs_crossref_lookup(paper):
                continue
            meta = self._lookup_metadata(paper.title)
            if meta:
                if not paper.doi:
                    paper.doi = meta.get("doi")
                if not paper.journal_name:
                    paper.journal_name = meta.get("journal_name")
                publication_type = meta.get("publication_type")
                if publication_type and publication_type not in paper.publication_types:
                    paper.publication_types.append(publication_type)
                if paper.citation_count is None:
                    paper.citation_count = meta.get("citation_count")
                if not paper.peer_reviewed:
                    paper.peer_reviewed = _infer_peer_reviewed(
                        publication_type, paper.journal_name
                    )
            time.sleep(_REQUEST_DELAY_SEC)
        return papers

    def _lookup_metadata(self, title: str) -> dict[str, str | int | None] | None:
        """Query Crossref and select best metadata match by title similarity."""
        params: dict[str, str | int] = {
            "query.title": title,
            "rows": max(1, min(self._cfg.rows, 5)),
            "select": "DOI,title,container-title,type,is-referenced-by-count",
        }
        if self._mailto:
            params["mailto"] = self._mailto
        try:
            resp = self._session.get(
                CROSSREF_WORKS_URL, params=params, timeout=self._cfg.timeout_sec
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.debug("crossref lookup failed title=%s err=%s", title[:80], e)
            return None

        items = (resp.json().get("message", {}) or {}).get("items", []) or []
        best_item: dict | None = None
        best_similarity = 0.0
        normalized_target = _normalize_title(title)
        for item in items:
            doi = (item.get("DOI") or "").strip().lower()
            item_titles = item.get("title") or []
            if not doi or not item_titles:
                continue
            candidate_title = _normalize_title(str(item_titles[0]))
            similarity = SequenceMatcher(
                None, normalized_target, candidate_title
            ).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_item = item
        if best_item and best_similarity >= self._cfg.min_title_similarity:
            journal_list = best_item.get("container-title") or []
            journal_name = str(journal_list[0]).strip() if journal_list else ""
            publication_type = str(best_item.get("type") or "").strip()
            citation_count = best_item.get("is-referenced-by-count")
            return {
                "doi": str(best_item.get("DOI") or "").strip().lower() or None,
                "journal_name": journal_name or None,
                "publication_type": publication_type or None,
                "citation_count": citation_count
                if isinstance(citation_count, int)
                else None,
            }
        return None


def _normalize_title(value: str) -> str:
    """Normalize title text before fuzzy matching."""
    return _SPACE.sub(" ", value.lower()).strip()


def _infer_peer_reviewed(
    publication_type: str | None, journal_name: str | None
) -> bool:
    if journal_name:
        return True
    if not publication_type:
        return False
    return publication_type.replace("-", "").lower() in {
        "journalarticle",
        "reviewarticle",
    }


def _needs_crossref_lookup(paper: PaperRecord) -> bool:
    if not paper.title.strip():
        return False
    if not paper.doi:
        return True
    if not paper.journal_name:
        return True
    if paper.citation_count is None:
        return True
    if not paper.publication_types:
        return True
    if not paper.peer_reviewed:
        return True
    return False
