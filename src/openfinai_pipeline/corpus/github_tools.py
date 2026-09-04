import base64
import re
from typing import Any
from urllib.parse import quote

from openfinai_pipeline.corpus.evidence_common import (
    _USER_AGENT,
    normalize_github_repo,
    request_error_payload,
    trim_text,
)
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

_GITHUB_API_ROOT = "https://api.github.com"
_DEFAULT_TREE_LIMIT = 200
_MAX_TREE_LIMIT = 1000
_DEFAULT_PATH_SEARCH_LIMIT = 20
_MAX_FILE_CHARS = 12000


def as_dict_tool(
    name: str,
    description: str,
    args_schema: type[BaseModel],
    handler,
) -> BaseTool:
    """Wrap a ``handler(arguments: dict)`` function as a LangChain StructuredTool.

    Keeps existing handlers untouched while exposing a proper per-field schema
    to the LLM (instead of an opaque ``arguments`` parameter).
    """

    def _run(**kwargs):
        return handler(kwargs)

    _run.__name__ = name
    return StructuredTool.from_function(
        func=_run,
        name=name,
        description=description,
        args_schema=args_schema,
    )


class _RepoArgs(BaseModel):
    repo_url: str = Field(default="", description="https://github.com/owner/name URL of the repo")
    repo: str = Field(default="", description="owner/name identifier for the repo")


class _ListTreeArgs(_RepoArgs):
    ref: str = Field(default="", description="Branch, tag, or commit SHA")
    path_prefix: str = Field(default="", description="Optional path prefix filter")
    limit: int = Field(default=_DEFAULT_TREE_LIMIT, ge=1, le=_MAX_TREE_LIMIT)


class _FetchFileArgs(_RepoArgs):
    path: str = Field(description="File path inside the repo")
    ref: str = Field(default="", description="Branch, tag, or commit SHA")


class _SearchPathsArgs(_RepoArgs):
    query: str = Field(description="Path search tokens; whitespace and punctuation are split into case-insensitive matches")
    ref: str = Field(default="", description="Branch, tag, or commit SHA")
    limit: int = Field(default=_DEFAULT_PATH_SEARCH_LIMIT, ge=1, le=_MAX_TREE_LIMIT)


def build_github_evidence_tools() -> list[BaseTool]:
    return [
        as_dict_tool(
            "inspect_github_repo",
            (
                "Inspect a public GitHub repository and return normalized repo metadata, "
                "default branch, README hints, and release URLs."
            ),
            _RepoArgs,
            inspect_github_repo,
        ),
        as_dict_tool(
            "list_github_repo_tree",
            (
                "List paths in a public GitHub repository tree for a branch or ref, optionally filtered by path prefix."
            ),
            _ListTreeArgs,
            list_github_repo_tree,
        ),
        as_dict_tool(
            "fetch_github_file",
            (
                "Fetch a text file from a public GitHub repository by path and ref. Suitable for README, configs, scripts, and notebooks as text."
            ),
            _FetchFileArgs,
            fetch_github_file,
        ),
        as_dict_tool(
            "list_github_release_assets",
            (
                "List releases and downloadable release assets for a public GitHub repository."
            ),
            _RepoArgs,
            list_github_release_assets,
        ),
        as_dict_tool(
            "search_github_repo_paths",
            (
                "Search paths in a public GitHub repository tree using client-side path matching. Useful for README, data, download, notebook, and config discovery."
            ),
            _SearchPathsArgs,
            search_github_repo_paths,
        ),
    ]


def inspect_github_repo(arguments: dict[str, object]) -> dict[str, object]:
    resolved = _resolve_repo(arguments)
    if "error" in resolved:
        return resolved
    repo = str(resolved["repo"])
    repo_url = str(resolved["repo_url"])
    metadata = _fetch_repo_metadata(repo, repo_url)
    if "error" in metadata:
        return metadata
    default_branch = str(metadata.get("default_branch", "")).strip()
    readme = _fetch_repo_readme_metadata(repo, repo_url)
    payload = {
        "repo": repo,
        "repo_url": repo_url,
        "default_branch": default_branch,
        "description": metadata.get("description", ""),
        "homepage": metadata.get("homepage", ""),
        "topics": metadata.get("topics", []),
        "html_url": metadata.get("html_url", repo_url),
        "releases_url": f"{repo_url}/releases",
        "api_url": f"{_GITHUB_API_ROOT}/repos/{repo}",
    }
    if "error" not in readme:
        payload.update(
            {
                "readme_path": readme.get("path", ""),
                "readme_html_url": readme.get("html_url", ""),
                "readme_download_url": readme.get("download_url", ""),
            }
        )
    return payload


def list_github_repo_tree(arguments: dict[str, object]) -> dict[str, object]:
    resolved = _resolve_repo(arguments)
    if "error" in resolved:
        return resolved
    repo = str(resolved["repo"])
    repo_url = str(resolved["repo_url"])
    metadata = _fetch_repo_metadata(repo, repo_url)
    if "error" in metadata:
        return metadata
    ref = str(arguments.get("ref", "")).strip() or str(metadata.get("default_branch", "")).strip()
    if not ref:
        return {"repo": repo, "repo_url": repo_url, "error": "missing_ref", "paths": []}
    payload = _fetch_repo_tree(repo, repo_url, ref=ref)
    if "error" in payload:
        return payload
    path_prefix = str(arguments.get("path_prefix", "")).strip().strip("/")
    limit = _clamp_limit(arguments.get("limit"), default=_DEFAULT_TREE_LIMIT)
    paths = list(payload.get("paths", []))
    if path_prefix:
        paths = [item for item in paths if str(item.get("path", "")).startswith(path_prefix)]
    return {
        "repo": repo,
        "repo_url": repo_url,
        "ref": ref,
        "path_prefix": path_prefix,
        "truncated": len(paths) > limit,
        "paths": paths[:limit],
    }


def fetch_github_file(arguments: dict[str, object]) -> dict[str, object]:
    import requests

    resolved = _resolve_repo(arguments)
    if "error" in resolved:
        return resolved
    repo = str(resolved["repo"])
    repo_url = str(resolved["repo_url"])
    path = str(arguments.get("path", "")).strip().lstrip("/")
    if not path:
        return {"repo": repo, "repo_url": repo_url, "error": "missing_path"}
    ref = str(arguments.get("ref", "")).strip()
    api_url = f"{_GITHUB_API_ROOT}/repos/{repo}/contents/{quote(path)}"
    request_url = f"{api_url}?ref={quote(ref)}" if ref else api_url
    try:
        response = requests.get(
            request_url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/vnd.github+json",
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        return request_error_payload(
            url=request_url,
            error=exc,
            repo=repo,
            repo_url=repo_url,
            path=path,
        )
    if response.status_code >= 400:
        return {
            "repo": repo,
            "repo_url": repo_url,
            "path": path,
            "ref": ref,
            "url": request_url,
            "error": f"http_error:{response.status_code}",
            "status_code": response.status_code,
            "response_snippet": trim_text(response.text),
        }
    try:
        payload = response.json()
    except ValueError:
        return {
            "repo": repo,
            "repo_url": repo_url,
            "path": path,
            "ref": ref,
            "url": request_url,
            "error": "invalid_json",
            "response_snippet": trim_text(response.text),
        }
    if not isinstance(payload, dict):
        return {
            "repo": repo,
            "repo_url": repo_url,
            "path": path,
            "ref": ref,
            "url": request_url,
            "error": "unexpected_payload",
        }
    file_type = str(payload.get("type", "")).strip()
    if file_type != "file":
        return {
            "repo": repo,
            "repo_url": repo_url,
            "path": path,
            "ref": ref,
            "url": request_url,
            "error": "unsupported_content",
            "content_type": file_type,
        }
    text, text_error = _decode_github_file_text(payload)
    result = {
        "repo": repo,
        "repo_url": repo_url,
        "path": path,
        "ref": ref or str(payload.get("sha", "")),
        "html_url": str(payload.get("html_url", "")),
        "download_url": str(payload.get("download_url", "")),
        "sha": str(payload.get("sha", "")),
        "size": int(payload.get("size", 0) or 0),
        "encoding": str(payload.get("encoding", "")),
        "text": text,
    }
    if text_error:
        result["error"] = text_error
    return result


def list_github_release_assets(arguments: dict[str, object]) -> dict[str, object]:
    import requests

    resolved = _resolve_repo(arguments)
    if "error" in resolved:
        return resolved
    repo = str(resolved["repo"])
    repo_url = str(resolved["repo_url"])
    url = f"{_GITHUB_API_ROOT}/repos/{repo}/releases"
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/vnd.github+json",
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        return request_error_payload(url=url, error=exc, repo=repo, repo_url=repo_url)
    if response.status_code >= 400:
        return {
            "repo": repo,
            "repo_url": repo_url,
            "url": url,
            "error": f"http_error:{response.status_code}",
            "status_code": response.status_code,
            "response_snippet": trim_text(response.text),
            "releases": [],
        }
    try:
        payload = response.json()
    except ValueError:
        return {
            "repo": repo,
            "repo_url": repo_url,
            "url": url,
            "error": "invalid_json",
            "releases": [],
            "response_snippet": trim_text(response.text),
        }
    if not isinstance(payload, list):
        return {
            "repo": repo,
            "repo_url": repo_url,
            "url": url,
            "error": "unexpected_payload",
            "releases": [],
        }
    releases: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        releases.append(
            {
                "name": str(item.get("name", "")),
                "tag_name": str(item.get("tag_name", "")),
                "html_url": str(item.get("html_url", "")),
                "draft": bool(item.get("draft", False)),
                "prerelease": bool(item.get("prerelease", False)),
                "assets": [
                    {
                        "name": str(asset.get("name", "")),
                        "browser_download_url": str(asset.get("browser_download_url", "")),
                        "content_type": str(asset.get("content_type", "")),
                        "size": int(asset.get("size", 0) or 0),
                    }
                    for asset in item.get("assets", [])
                    if isinstance(asset, dict)
                ],
            }
        )
    return {"repo": repo, "repo_url": repo_url, "url": url, "releases": releases}


def search_github_repo_paths(arguments: dict[str, object]) -> dict[str, object]:
    resolved = _resolve_repo(arguments)
    if "error" in resolved:
        return resolved
    repo = str(resolved["repo"])
    repo_url = str(resolved["repo_url"])
    query = str(arguments.get("query", "")).strip()
    if not query:
        return {"repo": repo, "repo_url": repo_url, "query": query, "matches": [], "error": "empty_query"}
    metadata = _fetch_repo_metadata(repo, repo_url)
    if "error" in metadata:
        return metadata
    ref = str(arguments.get("ref", "")).strip() or str(metadata.get("default_branch", "")).strip()
    if not ref:
        return {"repo": repo, "repo_url": repo_url, "query": query, "matches": [], "error": "missing_ref"}
    tree_payload = _fetch_repo_tree(repo, repo_url, ref=ref)
    if "error" in tree_payload:
        return tree_payload
    tokens = [token for token in re.split(r"[^a-zA-Z0-9]+", query.lower()) if token]
    limit = _clamp_limit(arguments.get("limit"), default=_DEFAULT_PATH_SEARCH_LIMIT)
    ranked = []
    for item in tree_payload.get("paths", []):
        path = str(item.get("path", ""))
        lowered = path.lower()
        if not tokens:
            continue
        match_count = sum(token in lowered for token in tokens)
        if match_count <= 0:
            continue
        ranked.append(
            (
                int(match_count == len(tokens)),
                match_count,
                int(any(path.lower().endswith(suffix) for suffix in (".md", ".ipynb", ".py", ".yaml", ".yml", ".json"))),
                -len(path),
                path,
                item,
            )
        )
    ranked.sort(reverse=True)
    matches = []
    for _, _, _, _, _, item in ranked[:limit]:
        matches.append(item)
    return {
        "repo": repo,
        "repo_url": repo_url,
        "ref": ref,
        "query": query,
        "matches": matches,
    }


def _resolve_repo(arguments: dict[str, object]) -> dict[str, object]:
    raw = str(arguments.get("repo_url", "") or arguments.get("repo", "")).strip()
    resolved = normalize_github_repo(raw)
    if resolved is None:
        return {"error": "unsupported_repo", "repo": raw, "repo_url": raw}
    repo, repo_url = resolved
    return {"repo": repo, "repo_url": repo_url}


def _fetch_repo_metadata(repo: str, repo_url: str) -> dict[str, object]:
    import requests

    url = f"{_GITHUB_API_ROOT}/repos/{repo}"
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/vnd.github+json",
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        return request_error_payload(url=url, error=exc, repo=repo, repo_url=repo_url)
    if response.status_code >= 400:
        return {
            "repo": repo,
            "repo_url": repo_url,
            "url": url,
            "error": f"http_error:{response.status_code}",
            "status_code": response.status_code,
            "response_snippet": trim_text(response.text),
        }
    try:
        payload = response.json()
    except ValueError:
        return {
            "repo": repo,
            "repo_url": repo_url,
            "url": url,
            "error": "invalid_json",
            "response_snippet": trim_text(response.text),
        }
    if not isinstance(payload, dict):
        return {
            "repo": repo,
            "repo_url": repo_url,
            "url": url,
            "error": "unexpected_payload",
        }
    return {
        "repo": repo,
        "repo_url": repo_url,
        "url": url,
        "default_branch": str(payload.get("default_branch", "")),
        "description": str(payload.get("description", "") or ""),
        "homepage": str(payload.get("homepage", "") or ""),
        "topics": payload.get("topics", []) if isinstance(payload.get("topics", []), list) else [],
        "html_url": str(payload.get("html_url", repo_url)),
    }


def _fetch_repo_readme_metadata(repo: str, repo_url: str) -> dict[str, object]:
    import requests

    url = f"{_GITHUB_API_ROOT}/repos/{repo}/readme"
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/vnd.github+json",
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        return request_error_payload(url=url, error=exc, repo=repo, repo_url=repo_url)
    if response.status_code >= 400:
        return {
            "repo": repo,
            "repo_url": repo_url,
            "url": url,
            "error": f"http_error:{response.status_code}",
            "status_code": response.status_code,
            "response_snippet": trim_text(response.text),
        }
    try:
        payload = response.json()
    except ValueError:
        return {
            "repo": repo,
            "repo_url": repo_url,
            "url": url,
            "error": "invalid_json",
            "response_snippet": trim_text(response.text),
        }
    if not isinstance(payload, dict):
        return {
            "repo": repo,
            "repo_url": repo_url,
            "url": url,
            "error": "unexpected_payload",
        }
    return {
        "path": str(payload.get("path", "")),
        "html_url": str(payload.get("html_url", "")),
        "download_url": str(payload.get("download_url", "")),
    }


def _fetch_repo_tree(repo: str, repo_url: str, *, ref: str) -> dict[str, object]:
    import requests

    url = f"{_GITHUB_API_ROOT}/repos/{repo}/git/trees/{quote(ref, safe='')}?recursive=1"
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/vnd.github+json",
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        return request_error_payload(
            url=url,
            error=exc,
            repo=repo,
            repo_url=repo_url,
            ref=ref,
            paths=[],
        )
    if response.status_code >= 400:
        return {
            "repo": repo,
            "repo_url": repo_url,
            "ref": ref,
            "url": url,
            "error": f"http_error:{response.status_code}",
            "status_code": response.status_code,
            "response_snippet": trim_text(response.text),
            "paths": [],
        }
    try:
        payload = response.json()
    except ValueError:
        return {
            "repo": repo,
            "repo_url": repo_url,
            "ref": ref,
            "url": url,
            "error": "invalid_json",
            "response_snippet": trim_text(response.text),
            "paths": [],
        }
    if not isinstance(payload, dict):
        return {
            "repo": repo,
            "repo_url": repo_url,
            "ref": ref,
            "url": url,
            "error": "unexpected_payload",
            "paths": [],
        }
    tree = payload.get("tree", [])
    paths: list[dict[str, object]] = []
    if isinstance(tree, list):
        for item in tree:
            if not isinstance(item, dict):
                continue
            paths.append(
                {
                    "path": str(item.get("path", "")),
                    "type": str(item.get("type", "")),
                    "sha": str(item.get("sha", "")),
                    "size": int(item.get("size", 0) or 0),
                    "url": str(item.get("url", "")),
                }
            )
    return {"repo": repo, "repo_url": repo_url, "ref": ref, "paths": paths}


def _decode_github_file_text(payload: dict[str, Any]) -> tuple[str, str]:
    content = str(payload.get("content", ""))
    encoding = str(payload.get("encoding", "")).strip()
    if content and encoding == "base64":
        try:
            decoded = base64.b64decode(content.encode("utf-8"), validate=False)
            return decoded.decode("utf-8", errors="replace")[:_MAX_FILE_CHARS], ""
        except Exception:
            return "", "decode_error"
    download_url = str(payload.get("download_url", "")).strip()
    if download_url:
        import requests

        try:
            response = requests.get(
                download_url,
                headers={"User-Agent": _USER_AGENT},
                timeout=20,
            )
        except requests.RequestException as exc:
            return "", f"download_error:{exc.__class__.__name__}"
        if response.status_code >= 400:
            return "", f"http_error:{response.status_code}"
        return response.text[:_MAX_FILE_CHARS], ""
    return "", "unsupported_content"


def _clamp_limit(value: object, *, default: int) -> int:
    try:
        limit = int(value if value is not None else default)
    except (TypeError, ValueError):
        limit = default
    return max(1, min(limit, _MAX_TREE_LIMIT))
