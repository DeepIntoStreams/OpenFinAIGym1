import hashlib
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from openfinai_pipeline.settings import DownloadConfig


class PDFDownloader:
    """Download PDFs, validate content, and optionally extract first-page text."""

    def __init__(self, cfg: DownloadConfig) -> None:
        self._cfg = cfg
        self._session = requests.Session()

    def download(
        self,
        paper_id: str,
        pdf_url: str,
    ) -> tuple[str, int, str, str] | None:
        """Download one paper PDF to a temporary file; return file metadata on success."""
        if not pdf_url:
            return None

        tmp_path: str | None = None
        try:
            resolved_url = self._resolve_pdf_url(pdf_url)
            resp = self._session.get(
                resolved_url,
                timeout=self._cfg.timeout_sec,
                stream=True,
                headers={
                    "Accept": "application/pdf,*/*;q=0.8",
                    "User-Agent": "fin-pipeline/0.1",
                },
            )
            resp.raise_for_status()

            size_limit_bytes = self._cfg.max_mb * 1024 * 1024
            written = 0
            hasher = hashlib.sha256()

            first_chunk = b""
            with tempfile.NamedTemporaryFile(
                prefix=f"fin_pipeline_{paper_id}_",
                suffix=".pdf",
                delete=False,
            ) as tmp:
                tmp_path = tmp.name
                for chunk in resp.iter_content(chunk_size=1024 * 32):
                    if not chunk:
                        continue
                    if not first_chunk:
                        first_chunk = chunk
                    written += len(chunk)
                    if written > size_limit_bytes:
                        raise ValueError("pdf too large")
                    tmp.write(chunk)
                    hasher.update(chunk)
            if not tmp_path:
                raise ValueError("empty tmp path")
            path = Path(tmp_path)
            if (first_chunk[:5] != b"%PDF-") or (not _is_valid_pdf(path)):
                raise ValueError("invalid pdf")
            return (
                str(path),
                written,
                hasher.hexdigest(),
                datetime.now(tz=timezone.utc).isoformat(),
            )
        except (requests.RequestException, OSError, ValueError):
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
            return None

    def _resolve_pdf_url(self, url: str) -> str:
        """Resolve DOI links to direct PDF-like targets when possible."""
        parsed = urlparse(url)
        if parsed.netloc.lower() not in {"doi.org", "www.doi.org"}:
            return url
        try:
            head = self._session.get(
                url,
                timeout=self._cfg.timeout_sec,
                allow_redirects=True,
                headers={"Accept": "application/pdf", "User-Agent": "fin-pipeline/0.1"},
            )
            ctype = (head.headers.get("Content-Type") or "").lower()
            if "application/pdf" in ctype and head.url:
                return head.url
            html = head.text or ""
            candidate = _extract_pdf_link_from_html(html, base_url=head.url or url)
            return candidate or (head.url or url)
        except requests.RequestException:
            return url

    def extract_excerpt(self, pdf_url: str, max_chars: int = 80000) -> str | None:
        """Fetch remote PDF and extract a short text excerpt from first pages."""
        if not pdf_url:
            return None
        resolved_url = self._resolve_pdf_url(pdf_url)
        try:
            resp = self._session.get(
                resolved_url,
                timeout=self._cfg.timeout_sec,
                stream=True,
                headers={
                    "Accept": "application/pdf,*/*;q=0.8",
                    "User-Agent": "fin-pipeline/0.1",
                },
            )
            resp.raise_for_status()
        except requests.RequestException:
            return None

        size_limit_bytes = self._cfg.max_mb * 1024 * 1024
        written = 0
        first_chunk = b""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
                for chunk in resp.iter_content(chunk_size=1024 * 32):
                    if not chunk:
                        continue
                    if not first_chunk:
                        first_chunk = chunk
                    written += len(chunk)
                    if written > size_limit_bytes:
                        return None
                    tmp.write(chunk)
            if first_chunk[:5] != b"%PDF-":
                return None
            return self.extract_excerpt_from_path(tmp_path, max_chars=max_chars)
        except (OSError, subprocess.SubprocessError):
            return None
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def extract_excerpt_from_path(
        self, pdf_path: str | Path, max_chars: int = 80000
    ) -> str | None:
        path = Path(pdf_path)
        if not path.exists() or not path.is_file() or not _is_valid_pdf(path):
            return None
        try:
            proc = subprocess.run(
                ["pdftotext", "-f", "1", "-nopgbrk", str(path), "-"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        text = (proc.stdout or "").strip()
        if not text:
            return None
        return text[:max_chars]

    def extract_page_texts_from_path(
        self,
        pdf_path: str | Path,
        *,
        max_pages: int | None = None,
        timeout_sec: int = 60,
    ) -> list[str] | None:
        path = Path(pdf_path)
        if not path.exists() or not path.is_file() or not _is_valid_pdf(path):
            return None
        cmd = ["pdftotext", "-f", "1"]
        if max_pages is not None and max_pages > 0:
            cmd.extend(["-l", str(max_pages)])
        cmd.extend([str(path), "-"])
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        raw = proc.stdout or ""
        if not raw.strip():
            return None
        pages = [page.strip() for page in raw.split("\f")]
        while pages and not pages[-1]:
            pages.pop()
        return pages or None


def _is_valid_pdf(path: Path) -> bool:
    """Quick PDF header check."""
    try:
        with path.open("rb") as f:
            head = f.read(5)
        return head == b"%PDF-"
    except OSError:
        return False


def _extract_pdf_link_from_html(html: str, base_url: str) -> str | None:
    """Find likely PDF URL in publisher landing page HTML."""
    m = re.search(
        r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        flags=re.IGNORECASE,
    )
    if m:
        return urljoin(base_url, m.group(1))
    m2 = re.search(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', html, flags=re.IGNORECASE)
    if m2:
        return urljoin(base_url, m2.group(1))
    return None
