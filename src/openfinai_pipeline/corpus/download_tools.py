import json
import re
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import quote_plus

from openfinai_pipeline.corpus.evidence_common import (
    LinkExtractor,
    TextExtractor,
    _USER_AGENT,
    find_snippet_after,
    request_error_payload,
    resolve_search_result_url,
    strip_html,
    trim_text,
)
from openfinai_pipeline.corpus.github_tools import (
    as_dict_tool,
    build_github_evidence_tools,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_PLAYWRIGHT_INSTALL_TIMEOUT_SEC = 300


class _SearchDownloadArgs(BaseModel):
    query: str = Field(description="Search query for dataset download evidence")
    limit: int = Field(default=3, ge=1, le=5)


class _UrlArgs(BaseModel):
    url: str = Field(description="Full http(s) URL to fetch or probe")


class _GdriveArgs(BaseModel):
    folder_id_or_url: str = Field(
        description=(
            "A Google Drive folder URL (https://drive.google.com/drive/folders/<ID>...), "
            "a file URL (https://drive.google.com/file/d/<ID>/...), a bare folder ID, "
            "or a bare file ID. The tool auto-detects which kind it is by probing "
            "the folder URL first and falling back to the file viewer URL."
        )
    )
    limit: int = Field(
        default=200,
        ge=1,
        le=1000,
        description="Maximum number of file entries to return when listing a folder.",
    )


def build_download_evidence_tools() -> list[BaseTool]:
    tools = [
        as_dict_tool(
            "search_download_evidence",
            (
                "Search the public web for official dataset download pages, repositories, API docs, or "
                "data-hosting pages relevant to the requested dataset. Returns structured errors instead of raising."
            ),
            _SearchDownloadArgs,
            search_download_evidence,
        ),
        as_dict_tool(
            "fetch_page_text",
            (
                "Fetch a public HTTP/HTTPS page and return cleaned text snippets plus extracted links so you can "
                "verify exact download URLs, repo structure, access restrictions, or dataset instructions. "
                "Returns structured errors instead of raising."
            ),
            _UrlArgs,
            fetch_page_text,
        ),
        *build_github_evidence_tools(),
        as_dict_tool(
            "probe_download_url",
            (
                "Probe a candidate HTTP/HTTPS download URL with HEAD or a bounded GET and report final URL, status "
                "code, content type, content disposition, and whether it looks like a real downloadable asset or "
                "just an HTML landing page."
            ),
            _UrlArgs,
            probe_download_url,
        ),
        as_dict_tool(
            "list_gdrive_folder",
            (
                "Enumerate the contents of a public Google Drive folder, or fetch metadata for a single public "
                "Google Drive file, without authentication. Accepts a folder URL, a file URL, or a bare ID. "
                "Returns per-file id/name/mime_type so the generated download script can call "
                "`gdown.download(id=...)` against evidenced IDs. Use this whenever paper context, README, or any "
                "tool result cites a drive.google.com folder or file link. Returns structured errors instead of raising."
            ),
            _GdriveArgs,
            list_gdrive_folder,
        ),
    ]
    if _playwright_available():
        tools.append(
            as_dict_tool(
                "fetch_page_rendered",
                (
                    "Render a public HTTP/HTTPS page in a headless browser and return normalized title, visible text, "
                    "and extracted links. Use this only after plain fetch/search still leave the page ambiguous or "
                    "JS-gated."
                ),
                _UrlArgs,
                fetch_page_rendered,
            )
        )
    return tools


def search_download_evidence(arguments: dict[str, object]) -> dict[str, object]:
    import requests

    query = str(arguments.get("query", "")).strip()
    limit = int(arguments.get("limit", 5) or 5)
    limit = max(1, min(limit, 5))
    if not query:
        return {"query": query, "results": [], "error": "empty_query"}

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=15)
    except requests.RequestException as exc:
        return request_error_payload(url=url, error=exc, query=query, results=[])

    if response.status_code >= 400:
        return {
            "query": query,
            "results": [],
            "error": f"http_error:{response.status_code}",
            "status_code": response.status_code,
            "url": url,
            "response_snippet": trim_text(response.text),
        }

    html = response.text
    pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    results: list[dict[str, str]] = []
    for match in pattern.finditer(html):
        href = resolve_search_result_url(match.group("href"))
        title = strip_html(match.group("title"))
        if not href or not title:
            continue
        snippet = find_snippet_after(html, match.end())
        results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= limit:
            break
    return {
        "query": query,
        "results": results,
        "url": url,
        "status_code": response.status_code,
    }


def fetch_page_text(arguments: dict[str, object]) -> dict[str, object]:
    import requests

    url = str(arguments.get("url", "")).strip()
    if not url.startswith(("http://", "https://")):
        return {"url": url, "error": "unsupported_url", "links": [], "text": ""}

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.1",
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        payload = request_error_payload(url=url, error=exc, links=[], text="")
        payload.setdefault("content_type", "")
        payload.setdefault("title", "")
        return payload

    content_type = response.headers.get("Content-Type", "")
    raw_text = response.text
    title = ""
    clean_text = ""
    links: list[dict[str, str]] = []
    if "html" in content_type.lower():
        title_match = re.search(
            r"<title[^>]*>(.*?)</title>", raw_text, re.IGNORECASE | re.DOTALL
        )
        if title_match:
            title = strip_html(title_match.group(1))
        extractor = TextExtractor()
        extractor.feed(raw_text)
        clean_text = extractor.text()
        link_extractor = LinkExtractor(url)
        link_extractor.feed(raw_text)
        links = link_extractor.links()
    else:
        clean_text = raw_text
        if "json" in content_type.lower():
            try:
                parsed = json.loads(raw_text)
                clean_text = json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
    clean_text = trim_text(clean_text)
    payload = {
        "url": url,
        "title": title,
        "content_type": content_type,
        "text": clean_text,
        "status_code": response.status_code,
        "links": links,
    }
    if response.status_code >= 400:
        payload["error"] = f"http_error:{response.status_code}"
    return payload


def probe_download_url(arguments: dict[str, object]) -> dict[str, object]:
    import requests

    url = str(arguments.get("url", "")).strip()
    if not url.startswith(("http://", "https://")):
        return {"url": url, "error": "unsupported_url"}

    headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    methods = ("head", "get")
    last_response = None
    for method in methods:
        try:
            response = requests.request(
                method.upper(),
                url,
                headers=headers,
                timeout=20,
                allow_redirects=True,
                stream=(method == "get"),
            )
        except requests.RequestException as exc:
            if method == methods[-1]:
                payload = request_error_payload(url=url, error=exc)
                payload.setdefault("final_url", url)
                payload.setdefault("content_type", "")
                payload.setdefault("content_disposition", "")
                payload.setdefault("response_snippet", "")
                payload["is_probable_download"] = False
                return payload
            continue
        last_response = response
        content_type = response.headers.get("Content-Type", "")
        content_disposition = response.headers.get("Content-Disposition", "")
        payload = {
            "url": url,
            "final_url": str(response.url),
            "status_code": response.status_code,
            "content_type": content_type,
            "content_disposition": content_disposition,
            "method": method.upper(),
            "is_probable_download": _looks_like_download(
                str(response.url),
                content_type,
                content_disposition,
            ),
        }
        if method == "get":
            try:
                snippet = response.text[:1500]
            except Exception:
                snippet = ""
            payload["response_snippet"] = trim_text(snippet)
        else:
            payload["response_snippet"] = ""
        if response.status_code >= 400:
            payload["error"] = f"http_error:{response.status_code}"
        if method == "head" and (
            response.status_code >= 400 or _looks_htmlish(content_type)
        ):
            continue
        return payload

    if last_response is None:
        return {"url": url, "error": "probe_failed", "is_probable_download": False}
    return {
        "url": url,
        "final_url": str(last_response.url),
        "status_code": last_response.status_code,
        "content_type": last_response.headers.get("Content-Type", ""),
        "content_disposition": last_response.headers.get("Content-Disposition", ""),
        "response_snippet": "",
        "is_probable_download": _looks_like_download(
            str(last_response.url),
            last_response.headers.get("Content-Type", ""),
            last_response.headers.get("Content-Disposition", ""),
        ),
    }


_GDRIVE_FOLDER_URL_RE = re.compile(
    r"drive\.google\.com/drive/(?:u/\d+/)?folders/([A-Za-z0-9_-]{15,})"
)
_GDRIVE_FILE_URL_RE = re.compile(
    r"drive\.google\.com/file/(?:u/\d+/)?d/([A-Za-z0-9_-]{15,})"
)
_GDRIVE_OPEN_URL_RE = re.compile(
    r"drive\.google\.com/open\?(?:[^#]*&)?id=([A-Za-z0-9_-]{15,})"
)
_GDRIVE_UC_URL_RE = re.compile(
    r"drive\.google\.com/uc\?(?:[^#]*&)?id=([A-Za-z0-9_-]{15,})"
)
_GDRIVE_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{15,}$")
_GDRIVE_FOLDER_DATA_RE = re.compile(
    r"window\['_DRIVE_ivd'\]\s*=\s*'(.+?)';", re.DOTALL
)
_GDRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
_GDRIVE_TITLE_SUFFIX = " - Google Drive"


def _resolve_gdrive_id(value: str) -> tuple[str, str]:
    """Extract a Drive ID and a kind hint from a folder URL, file URL, or bare ID.

    Returns ``(resolved_id, kind_hint)`` where ``kind_hint`` is one of
    ``"folder"``, ``"file"``, or ``"unknown"``. An empty ``resolved_id`` means
    the input did not look like any recognized Drive form.
    """
    raw = (value or "").strip()
    if not raw:
        return "", "unknown"
    folder_match = _GDRIVE_FOLDER_URL_RE.search(raw)
    if folder_match:
        return folder_match.group(1), "folder"
    file_match = _GDRIVE_FILE_URL_RE.search(raw)
    if file_match:
        return file_match.group(1), "file"
    open_match = _GDRIVE_OPEN_URL_RE.search(raw) or _GDRIVE_UC_URL_RE.search(raw)
    if open_match:
        return open_match.group(1), "unknown"
    if _GDRIVE_BARE_ID_RE.match(raw):
        return raw, "unknown"
    return "", "unknown"


def _parse_drive_folder_html(html: str) -> list[dict[str, object]] | None:
    """Extract file metadata from a Drive folder page's embedded data block.

    Mirrors gdown's ``_DRIVE_ivd`` parsing approach. Returns ``None`` when the
    data block is missing or unparseable; returns an empty list when the block
    parses but contains no entries.
    """
    match = _GDRIVE_FOLDER_DATA_RE.search(html)
    if not match:
        return None
    try:
        decoded = bytes(match.group(1), "utf-8").decode("unicode_escape")
        folder_arr = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(folder_arr, list) or not folder_arr:
        return []
    contents = folder_arr[0] if folder_arr[0] is not None else []
    if not isinstance(contents, list):
        return []
    files: list[dict[str, object]] = []
    for entry in contents:
        if not isinstance(entry, list) or len(entry) < 4:
            continue
        file_id = str(entry[0] or "")
        if not file_id:
            continue
        name = str(entry[2] or "") if len(entry) > 2 else ""
        mime = str(entry[3] or "") if len(entry) > 3 else ""
        files.append(
            {
                "id": file_id,
                "name": name,
                "mime_type": mime,
                "is_folder": mime == _GDRIVE_FOLDER_MIME,
            }
        )
    return files


def _parse_drive_file_title(html: str) -> str:
    """Extract the filename from a Drive file viewer page's <title> tag."""
    match = re.search(r"<title>([^<]*?)</title>", html, re.IGNORECASE)
    if not match:
        return ""
    title = strip_html(match.group(1)).strip()
    if title.endswith(_GDRIVE_TITLE_SUFFIX):
        title = title[: -len(_GDRIVE_TITLE_SUFFIX)].strip()
    return title


def list_gdrive_folder(arguments: dict[str, object]) -> dict[str, object]:
    """Enumerate a public Google Drive folder, or fetch metadata for one file.

    Probes the folder URL first when the input could be either kind; falls
    back to the file viewer URL when the folder data block is absent. Never
    raises — returns a structured ``error`` field on every failure path.
    """
    import requests

    raw_input = str(arguments.get("folder_id_or_url", "")).strip()
    limit = int(arguments.get("limit", 200) or 200)
    limit = max(1, min(limit, 1000))

    if not raw_input:
        return {
            "input": raw_input,
            "resolved_id": "",
            "kind": "unknown",
            "files": [],
            "total_found": 0,
            "truncated": False,
            "error": "empty_input",
        }

    resolved_id, kind_hint = _resolve_gdrive_id(raw_input)
    if not resolved_id:
        return {
            "input": raw_input,
            "resolved_id": "",
            "kind": "unknown",
            "files": [],
            "total_found": 0,
            "truncated": False,
            "error": "unsupported_input",
        }

    headers = {"User-Agent": _USER_AGENT, "Accept": "text/html,*/*;q=0.5"}

    # Try folder URL first when caller hinted folder, or kind is unknown.
    if kind_hint in ("folder", "unknown"):
        folder_url = f"https://drive.google.com/drive/folders/{resolved_id}"
        try:
            response = requests.get(folder_url, headers=headers, timeout=20)
        except requests.RequestException as exc:
            return request_error_payload(
                url=folder_url,
                error=exc,
                input=raw_input,
                resolved_id=resolved_id,
                kind="folder" if kind_hint == "folder" else "unknown",
                files=[],
                total_found=0,
                truncated=False,
            )
        if response.status_code < 400:
            files = _parse_drive_folder_html(response.text)
            if files is not None:
                truncated = len(files) > limit
                return {
                    "input": raw_input,
                    "resolved_id": resolved_id,
                    "kind": "folder",
                    "folder_url": folder_url,
                    "files": files[:limit],
                    "total_found": len(files),
                    "truncated": truncated,
                    "status_code": response.status_code,
                }
            if kind_hint == "folder":
                return {
                    "input": raw_input,
                    "resolved_id": resolved_id,
                    "kind": "folder",
                    "folder_url": folder_url,
                    "files": [],
                    "total_found": 0,
                    "truncated": False,
                    "error": "parse_failed",
                    "status_code": response.status_code,
                    "response_snippet": trim_text(response.text),
                }
            # Unknown hint: fall through to file URL probe.
        elif kind_hint == "folder":
            return {
                "input": raw_input,
                "resolved_id": resolved_id,
                "kind": "folder",
                "folder_url": folder_url,
                "files": [],
                "total_found": 0,
                "truncated": False,
                "error": f"http_error:{response.status_code}",
                "status_code": response.status_code,
            }

    # Single-file viewer.
    file_url = f"https://drive.google.com/file/d/{resolved_id}/view"
    try:
        response = requests.get(file_url, headers=headers, timeout=20)
    except requests.RequestException as exc:
        return request_error_payload(
            url=file_url,
            error=exc,
            input=raw_input,
            resolved_id=resolved_id,
            kind="file",
            files=[],
            total_found=0,
            truncated=False,
        )
    if response.status_code >= 400:
        return {
            "input": raw_input,
            "resolved_id": resolved_id,
            "kind": "file",
            "file_url": file_url,
            "files": [],
            "total_found": 0,
            "truncated": False,
            "error": f"http_error:{response.status_code}",
            "status_code": response.status_code,
        }
    name = _parse_drive_file_title(response.text)
    return {
        "input": raw_input,
        "resolved_id": resolved_id,
        "kind": "file",
        "file_url": file_url,
        "files": [
            {"id": resolved_id, "name": name, "mime_type": "", "is_folder": False}
        ],
        "total_found": 1,
        "truncated": False,
        "status_code": response.status_code,
    }


def fetch_page_rendered(arguments: dict[str, object]) -> dict[str, object]:
    url = str(arguments.get("url", "")).strip()
    if not url.startswith(("http://", "https://")):
        return {"url": url, "error": "unsupported_url", "links": [], "text": ""}
    if not _playwright_available():
        return {"url": url, "error": "playwright_unavailable", "links": [], "text": ""}

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except Exception:
        return {"url": url, "error": "playwright_unavailable", "links": [], "text": ""}

    browser_ready, install_error = _ensure_playwright_chromium_installed(
        sync_playwright
    )
    if not browser_ready:
        return {
            "url": url,
            "error": "playwright_browser_unavailable",
            "message": install_error or "Chromium is not installed for Playwright.",
            "links": [],
            "text": "",
            "title": "",
        }

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(user_agent=_USER_AGENT)
            page.goto(url, wait_until="networkidle", timeout=20_000)
            html = page.content()
            final_url = page.url
            title = page.title()
            browser.close()
    except PlaywrightError as exc:
        return {
            "url": url,
            "error": f"playwright_error:{exc.__class__.__name__}",
            "message": str(exc),
            "links": [],
            "text": "",
            "title": "",
        }

    extractor = TextExtractor()
    extractor.feed(html)
    link_extractor = LinkExtractor(final_url)
    link_extractor.feed(html)
    return {
        "url": url,
        "final_url": final_url,
        "title": title,
        "content_type": "text/html; rendered",
        "text": trim_text(extractor.text()),
        "status_code": 200,
        "links": link_extractor.links(),
        "rendered": True,
    }


def _playwright_available() -> bool:
    return find_spec("playwright.sync_api") is not None


def _ensure_playwright_chromium_installed(sync_playwright) -> tuple[bool, str | None]:
    try:
        with sync_playwright() as playwright:
            executable_path = Path(playwright.chromium.executable_path)
            if executable_path.exists():
                return True, None
    except Exception as exc:
        return False, f"Failed to inspect the Playwright Chromium installation: {exc}"

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            timeout=_PLAYWRIGHT_INSTALL_TIMEOUT_SEC,
            check=False,
        )
    except Exception as exc:
        return False, f"Failed to run 'python -m playwright install chromium': {exc}"

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        details = stderr or stdout or f"exit code {completed.returncode}"
        return False, f"Failed to install Chromium for Playwright: {details}"

    try:
        with sync_playwright() as playwright:
            executable_path = Path(playwright.chromium.executable_path)
            if executable_path.exists():
                return True, None
    except Exception as exc:
        return False, f"Chromium installation completed but verification failed: {exc}"

    return False, (
        "Chromium installation completed but the browser executable was not found."
    )


def _looks_htmlish(content_type: str) -> bool:
    lowered = (content_type or "").lower()
    return "html" in lowered or "text/plain" in lowered or lowered == ""


def _looks_like_download(url: str, content_type: str, content_disposition: str) -> bool:
    lowered_type = (content_type or "").lower()
    lowered_disp = (content_disposition or "").lower()
    lowered_url = (url or "").lower()
    if "attachment" in lowered_disp or "filename=" in lowered_disp:
        return True
    if any(
        token in lowered_type
        for token in (
            "zip",
            "gzip",
            "csv",
            "excel",
            "octet-stream",
            "parquet",
            "jsonl",
            "binary",
        )
    ):
        return True
    return lowered_url.endswith(
        (".zip", ".csv", ".json", ".jsonl", ".parquet", ".gz", ".bz2", ".xz", ".tar", ".tgz")
    )
