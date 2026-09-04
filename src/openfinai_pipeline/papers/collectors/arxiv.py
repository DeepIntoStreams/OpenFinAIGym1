import hashlib
import logging
import math
import random
import time
import urllib.parse
from datetime import datetime, timedelta, timezone

import feedparser
import requests

from openfinai_pipeline.errors import ScraperError
from openfinai_pipeline.papers.schemas import PaperRecord, ScopeDefinition
from openfinai_pipeline.settings import ArxivConfig

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"


class ArxivClient:
    """arXiv collector with pagination, optional time slicing, and 429 backoff."""

    def __init__(self, cfg: ArxivConfig) -> None:
        self._cfg = cfg
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "fin-pipeline/0.1 (+arxiv client)"})
        # Monotonic timestamp of last outbound request, used to enforce a
        # global min-gap between requests regardless of which slice/query/page
        # initiated the call. 0.0 means "no request yet" → first call skips
        # the throttle.
        self._last_request_ts: float = 0.0

    def scrape_scope(
        self,
        scope: ScopeDefinition,
        since_date: datetime | None,
        until_date: datetime | None,
        max_papers: int,
    ) -> list[PaperRecord]:
        """Fetch one scope across queries, chronological slices, and pages.

        Slices run oldest to newest; each slice uses arXiv's configured
        descending result order. Collection stops at the global paper limit.
        """
        seen: dict[str, PaperRecord] = {}
        max_results = min(self._cfg.page_size, max_papers)
        pages_per_query = min(
            self._cfg.max_pages,
            max(1, math.ceil(max_papers / max(1, max_results))),
        )

        # The logged request count is an upper bound because collection may stop early.
        time_slices = _build_time_slices(
            since_date=since_date,
            until_date=until_date,
            slice_days=self._cfg.time_slice_days,
            enabled=self._cfg.enable_time_slicing,
        )
        if not time_slices:
            time_slices = [(since_date, until_date)]

        logger.info(
            "scope=%s arxiv_budget queries=%d slices=%d pages_per_query=%d max_requests=%d",
            scope.id,
            len(scope.queries),
            len(time_slices),
            pages_per_query,
            len(scope.queries) * len(time_slices) * pages_per_query,
        )

        # Interleave query vocabularies within each slice for balanced coverage.
        for slice_start, slice_end in time_slices:
            for q_idx, query in enumerate(scope.queries, start=1):
                for page in range(pages_per_query):
                    start = page * max_results
                    slice_label = (
                        f"{slice_start.strftime('%Y-%m-%d')}..{slice_end.strftime('%Y-%m-%d')}"
                        if slice_start and slice_end
                        else "any"
                    )
                    logger.info(
                        "scope=%s arxiv_request query=%d/%d slice=%s page=%d/%d",
                        scope.id,
                        q_idx,
                        len(scope.queries),
                        slice_label,
                        page + 1,
                        pages_per_query,
                    )
                    try:
                        page_papers = self._search_page(
                            query=query,
                            categories=scope.categories,
                            start=start,
                            max_results=max_results,
                            date_range=_to_arxiv_date_range(slice_start, slice_end),
                        )
                    except ScraperError as e:
                        logger.warning(
                            "scope=%s query=%s slice=%s..%s page=%d arxiv_page_failed=%s",
                            scope.id,
                            query[:80],
                            slice_start.isoformat() if slice_start else "None",
                            slice_end.isoformat() if slice_end else "None",
                            page,
                            e,
                        )
                        break
                    if not page_papers:
                        break

                    for paper in page_papers:
                        if since_date and paper.published_at:
                            published_dt = _parse_datetime(paper.published_at)
                            if published_dt and published_dt < since_date:
                                continue
                        if until_date and paper.published_at:
                            published_dt = _parse_datetime(paper.published_at)
                            if published_dt and published_dt > until_date:
                                continue

                        if scope.id not in paper.scope_ids:
                            paper.scope_ids.append(scope.id)
                        existing = seen.get(paper.paper_id)
                        if existing is None:
                            seen[paper.paper_id] = paper
                        elif scope.id not in existing.scope_ids:
                            existing.scope_ids.append(scope.id)

                        if len(seen) >= max_papers:
                            break

                    if len(seen) >= max_papers:
                        break
                    # No explicit sleep here — `_get_with_retry` throttles every
                    # outbound request globally so slice/query transitions are
                    # rate-limited too, not just within-slice page transitions.

                if len(seen) >= max_papers:
                    break
            if len(seen) >= max_papers:
                break

        logger.info("scope=%s scraped=%d", scope.id, len(seen))
        return list(seen.values())

    def query_recall_check(
        self, scope: ScopeDefinition, arxiv_id: str
    ) -> tuple[bool, str | None]:
        """Check whether a known arXiv ID can be retrieved by current queries."""
        for query in scope.queries:
            papers = self._search_page(
                query=query,
                categories=scope.categories,
                start=0,
                max_results=1,
                date_range=None,
                id_filter=arxiv_id,
            )
            if papers and any(p.paper_id == arxiv_id for p in papers):
                return True, query
        return False, None

    def _search_page(
        self,
        query: str,
        categories: list[str],
        start: int,
        max_results: int,
        date_range: str | None,
        id_filter: str | None = None,
    ) -> list[PaperRecord]:
        """Issue one arXiv API request and parse feed entries into paper records."""
        search_parts = [f"all:{query}"]
        if categories:
            cat_filter = " OR ".join(f"cat:{c}" for c in categories)
            search_parts.append(f"({cat_filter})")
        if date_range:
            search_parts.append(f"submittedDate:[{date_range}]")
        if id_filter:
            search_parts.append(f"id:{id_filter}")

        params = {
            "search_query": " AND ".join(search_parts),
            "start": start,
            "max_results": max_results,
            "sortBy": self._cfg.sort_by,
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

        resp = self._get_with_retry(url)

        feed = feedparser.parse(resp.text)
        papers: list[PaperRecord] = []
        for entry in feed.entries:
            paper = self._entry_to_paper(entry)
            if paper:
                papers.append(paper)
        return papers

    def _get_with_retry(self, url: str) -> requests.Response:
        """GET with global throttle, exponential backoff, and 429 handling.

        Throttling runs once per attempt so even backoff-then-retry honors the
        minimum inter-request interval. The 429 backoff floor is set well above
        the polite-interval floor because arXiv 429s are sticky: a short retry
        almost always triggers another 429.
        """
        max_attempts = 6
        for attempt in range(1, max_attempts + 1):
            self._throttle()
            try:
                resp = self._session.get(url, timeout=self._cfg.timeout_sec)
                self._last_request_ts = time.monotonic()
                status_code = int(getattr(resp, "status_code", 200))
                if status_code == 429:
                    wait = _retry_after_seconds(resp, attempt)
                    logger.warning(
                        "arXiv rate-limited (429), attempt=%d/%d sleep=%.1fs",
                        attempt,
                        max_attempts,
                        wait,
                    )
                    if attempt == max_attempts:
                        raise ScraperError(
                            f"arXiv request failed after retries: HTTP 429 for {url}"
                        )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                self._last_request_ts = time.monotonic()
                if attempt == max_attempts:
                    raise ScraperError(f"arXiv request failed: {e}") from e
                wait = min(30.0, 1.5 * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5))
                logger.warning(
                    "arXiv transient error, attempt=%d/%d sleep=%.1fs err=%s",
                    attempt,
                    max_attempts,
                    wait,
                    e,
                )
                time.sleep(wait)
        raise ScraperError("arXiv request failed: exhausted retries")

    def _throttle(self) -> None:
        """Block until at least ``request_interval_sec`` has passed since the
        last outbound request. Skipped on the very first call."""
        if self._last_request_ts <= 0.0:
            return
        elapsed = time.monotonic() - self._last_request_ts
        remaining = self._cfg.request_interval_sec - elapsed
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _entry_to_paper(entry: dict) -> PaperRecord | None:
        """Convert a feedparser entry into canonical PaperRecord."""
        raw_id = entry.get("id", "").split("/abs/")[-1]
        paper_id = raw_id.split("v")[0].strip()
        if not paper_id:
            return None

        pdf_url = None
        for link in entry.get("links", []):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break

        payload = {
            "id": paper_id,
            "title": entry.get("title", "").strip(),
            "summary": entry.get("summary", "").strip(),
            "published": entry.get("published"),
            "updated": entry.get("updated"),
        }
        journal_ref = _clean_optional_text(entry.get("arxiv_journal_ref"))

        return PaperRecord(
            paper_id=paper_id,
            title=entry.get("title", "").replace("\n", " ").strip(),
            abstract=entry.get("summary", "").replace("\n", " ").strip(),
            authors=[a.get("name", "") for a in entry.get("authors", [])],
            categories=[t.get("term", "") for t in entry.get("tags", [])],
            published_at=entry.get("published"),
            updated_at=entry.get("updated"),
            pdf_url=pdf_url,
            primary_category=entry.get("arxiv_primary_category", {}).get("term"),
            doi=(entry.get("arxiv_doi") or "").strip().lower() or None,
            journal_name=journal_ref,
            publication_types=["journal_reference"] if journal_ref else [],
            peer_reviewed=bool(journal_ref),
            raw_payload_hash=hashlib.sha256(repr(payload).encode("utf-8")).hexdigest(),
        )


def since_days_to_datetime(since_days: int | None) -> datetime | None:
    """Convert relative since_days to absolute UTC datetime."""
    if since_days is None:
        return None
    return datetime.now(tz=timezone.utc) - timedelta(days=since_days)


def _parse_datetime(value: str) -> datetime | None:
    """Parse arXiv timestamp format safely."""
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _to_arxiv_date_range(start: datetime | None, end: datetime | None) -> str | None:
    """Build arXiv submittedDate range clause."""
    if not start and not end:
        return None
    if start is None:
        start = datetime(1991, 1, 1, tzinfo=timezone.utc)
    if end is None:
        end = datetime.now(tz=timezone.utc)
    return f"{start.strftime('%Y%m%d')}0000 TO {end.strftime('%Y%m%d')}2359"


def _build_time_slices(
    since_date: datetime | None,
    until_date: datetime | None,
    slice_days: int,
    enabled: bool,
) -> list[tuple[datetime | None, datetime | None]]:
    """Split [since, until] into fixed-size date slices to reduce miss/dup risk."""
    if not enabled or since_date is None:
        return [(since_date, until_date)]
    end = until_date or datetime.now(tz=timezone.utc)
    if since_date > end:
        return [(since_date, end)]
    step = max(1, slice_days)
    slices: list[tuple[datetime, datetime]] = []
    cur_start = since_date
    while cur_start <= end:
        cur_end = min(end, cur_start + timedelta(days=step - 1))
        slices.append((cur_start, cur_end))
        cur_start = cur_end + timedelta(days=1)
    return slices


def _retry_after_seconds(resp: requests.Response, attempt: int) -> float:
    """Get retry wait from Retry-After header or fallback exponential backoff.

    arXiv 429s are sticky — a 2-3s retry almost always trips the limit again.
    Floors: 10s minimum when the server sends ``Retry-After`` (servers can
    under-report), 30s start when it doesn't. Sequence with no header is
    roughly 30s → 60s → 120s → 240s → 300s → 300s (cap), giving the limiter
    enough time to fully clear without abandoning the run too eagerly.
    """
    headers = getattr(resp, "headers", {}) or {}
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            return max(10.0, float(retry_after))
        except ValueError:
            pass
    return min(300.0, 30.0 * (2 ** (attempt - 1)) + random.uniform(0.0, 5.0))


def _clean_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None
