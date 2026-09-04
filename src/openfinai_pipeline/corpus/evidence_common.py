import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlparse

_USER_AGENT = (
    "OpenFinGymDatasetEvidence/1.0 (+https://github.com/DeepIntoStreams/OpenFinGym1)"
)
_MAX_FETCH_CHARS = 12000
_MAX_FETCH_LINKS = 40
_GITHUB_HOSTS = {"github.com", "www.github.com"}


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        clean = " ".join(data.split())
        if clean:
            self._chunks.append(clean)

    def text(self) -> str:
        return " ".join(self._chunks)


class LinkExtractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self._base_url = base_url
        self._current_href: str | None = None
        self._current_text: list[str] = []
        self._links: list[dict[str, str]] = []
        self._seen: set[tuple[str, str]] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        self._current_href = urljoin(self._base_url, href)
        self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is None:
            return
        clean = " ".join(data.split())
        if clean:
            self._current_text.append(clean)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._current_href is None:
            return
        href = self._current_href.strip()
        text = " ".join(self._current_text).strip()
        self._current_href = None
        self._current_text = []
        if not href or href.startswith(("javascript:", "mailto:", "tel:")):
            return
        if not href.startswith(("http://", "https://")):
            return
        key = (href, text)
        if key in self._seen:
            return
        self._seen.add(key)
        self._links.append({"url": href, "text": text})

    def links(self, limit: int = _MAX_FETCH_LINKS) -> list[dict[str, str]]:
        return self._links[:limit]


def request_error_payload(url: str, error: Exception, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "url": url,
        "error": f"request_exception:{error.__class__.__name__}",
        "message": str(error),
    }
    status_code = getattr(getattr(error, "response", None), "status_code", None)
    if status_code is not None:
        payload["status_code"] = status_code
    payload.update(extra)
    return payload


def trim_text(text: str) -> str:
    return " ".join((text or "").split())[:_MAX_FETCH_CHARS]


def strip_html(raw: str) -> str:
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", raw or "")).split())


def find_snippet_after(html: str, start: int) -> str:
    match = re.search(
        r"<a[^>]+result__snippet[^>]*>(?P<snippet>.*?)</a>",
        html[start:],
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        match = re.search(
            r"<div[^>]+result__snippet[^>]*>(?P<snippet>.*?)</div>",
            html[start:],
            re.IGNORECASE | re.DOTALL,
        )
    if not match:
        return ""
    return strip_html(match.group("snippet"))


def resolve_search_result_url(href: str) -> str:
    if href.startswith("//"):
        return f"https:{href}"
    if href.startswith("/l/?"):
        query = parse_qs(urlparse(href).query)
        uddg = query.get("uddg", [""])[0]
        return unescape(uddg)
    return href


def normalize_github_repo(value: str) -> tuple[str, str] | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    repo_match = re.fullmatch(r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", raw)
    if repo_match:
        owner, repo_name = repo_match.groups()
        repo_name = repo_name.removesuffix(".git")
        repo = f"{owner}/{repo_name}"
        return repo, f"https://github.com/{repo}"
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in _GITHUB_HOSTS:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    owner = parts[0]
    repo_name = parts[1].removesuffix(".git")
    if not owner or not repo_name:
        return None
    repo = f"{owner}/{repo_name}"
    return repo, f"https://github.com/{repo}"

