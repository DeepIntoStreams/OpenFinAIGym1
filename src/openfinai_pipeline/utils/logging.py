import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_PHASE_TAG_RE = re.compile(r"^\[phase[^\]]+\]\s*")


def _utf8_stream_handler() -> logging.StreamHandler:
    """StreamHandler that writes UTF-8 (avoids cp1252 errors on Windows)."""
    stream = sys.stderr
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return logging.StreamHandler(stream)


class _ResponseBodyFilter(logging.Filter):
    """Suppress full LLM response/tool body dumps from the console.

    Messages that contain ``[response]``, ``[input]``, or ``[result]``
    section markers are the verbose multi-line LLM dumps.  This filter
    keeps them out of the terminal at every tier; the file handler
    (which has no filter) continues to capture everything for
    post-mortem inspection.
    """

    _MARKERS = ("\n[response]\n", "\n[input]\n", "\n[result]\n")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(marker in msg for marker in self._MARKERS)


class _ThirdPartyConsoleFilter(logging.Filter):
    """Drop INFO-level records from noisy third-party HTTP clients.

    The OpenAI/OpenRouter/Anthropic SDKs all log every HTTP request via
    ``httpx`` at INFO level (e.g. ``HTTP Request: POST .../chat/completions``).
    A single dataset candidate can trigger 10-20 of these per round, which
    drowns out the per-candidate stage markers our pipeline emits.  This
    filter suppresses INFO records from those loggers on the console while
    still allowing WARNING+ through so genuine HTTP failures are visible.
    The file handler stays unfiltered, so forensics retain everything.
    """

    _NOISY_PREFIXES = (
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "anthropic",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        name = record.name or ""
        return not any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in self._NOISY_PREFIXES
        )


class _ConsoleTierFilter(logging.Filter):
    """Decide which records reach the terminal based on the verbosity tier.

    At default (``verbose=False``) only stage records (carrying
    ``extra={"console_stage": True}``) and WARNING+ records are shown.
    At ``verbose=True`` regular ``logger.info`` calls are shown too.
    """

    def __init__(self, *, verbose: bool) -> None:
        super().__init__()
        self._verbose = verbose

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        if getattr(record, "console_stage", False):
            return True
        return self._verbose


class _CompactConsoleFormatter(logging.Formatter):
    """Strip timestamp, level, and logger name from console output.

    The user reads console output live; the timestamped, fully-qualified
    format is preserved in the file log.  On the terminal we keep just
    the message, with a few stylistic touches:

    * Stage records (``console_stage=True``) print as-is.  The caller's
      leading ``[phaseX:Y]`` tag is preserved as the user's anchor.
    * Plain INFO records (only seen at ``-v``) get a 2-space indent and
      have any leading ``[phaseX:Y]`` tag stripped, since the most
      recent stage line already supplies that context.
    * WARNING and ERROR/CRITICAL records get an indent and a
      ``[WARN]`` / ``[ERROR]`` prefix so they remain visible while
      staying grouped under the most recent stage line.
    """

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f"  [ERROR] {self._strip_tag(msg)}"
        if record.levelno == logging.WARNING:
            return f"  [WARN]  {self._strip_tag(msg)}"
        if getattr(record, "console_stage", False):
            return msg
        return f"  {self._strip_tag(msg)}"

    @staticmethod
    def _strip_tag(msg: str) -> str:
        return _PHASE_TAG_RE.sub("", msg, count=1)


def configure_practice_logging(
    log_root: str,
    practice: str,
    run_tag: str | None = None,
    *,
    verbose: bool = False,
) -> Path:
    """Write logs to <log_root>/<run_tag>/<practice>.log and return that run directory.

    The file handler always captures the full timestamped record at
    INFO and above.  The console handler shows only stage markers,
    warnings, and errors at default; pass ``verbose=True`` (or
    ``--verbose`` on the CLI) to also surface plain INFO records on
    the terminal in compact, indented form.

    LLM response bodies (``[response]``/``[input]``/``[result]`` blocks)
    are *always* suppressed from the terminal regardless of verbosity
    and remain available in the file log only.
    """
    run_dir = Path(log_root) / run_tag if run_tag else Path(log_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{practice}.log"

    console = _utf8_stream_handler()
    console.addFilter(_ConsoleTierFilter(verbose=verbose))
    console.addFilter(_ResponseBodyFilter())
    console.addFilter(_ThirdPartyConsoleFilter())
    console.setFormatter(_CompactConsoleFormatter())

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console, file_handler],
        force=True,
    )
    return run_dir


def log_stage(
    logger: logging.Logger,
    msg: str,
    *args: object,
    level: int = logging.INFO,
) -> None:
    """Emit a record that is always shown on the console (a stage marker).

    Stage records bypass the default-tier console filter.  They are
    rendered without timestamp/level/logger-name on the terminal and
    are intended for high-level pipeline milestones such as phase
    boundaries, per-item progress headers, and end-of-phase tallies.
    Use the existing ``[phaseN:foo]`` tag at the start of *msg* so the
    user has a consistent visual anchor.
    """
    logger.log(level, msg, *args, extra={"console_stage": True})


def log_detail(
    logger: logging.Logger,
    msg: str,
    *args: object,
    level: int = logging.INFO,
) -> None:
    """Emit a record only to the file handler(s); never reaches the console.

    Use this for high-volume diagnostic output (full chunk lists,
    per-intent ranking dumps, prompt previews) that is valuable for
    post-mortem inspection but would bomb the terminal if shown live.
    """
    record = logger.makeRecord(
        name=logger.name,
        level=level,
        fn="",
        lno=0,
        msg=msg,
        args=args,
        exc_info=None,
    )
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.emit(record)


def fmt_bytes(n: int) -> str:
    """Format a byte count (1KB = 1024 B) compactly: ``X.YMB`` if >=1MB else ``KKB`` (min 1KB)."""
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f}MB"
    return f"{max(1, (n + 512) // 1024)}KB"


def fmt_chars(n: int) -> str:
    """Format a character count (1K = 1000 chars) compactly: ``X.YM`` if >=1M else ``NK`` else raw int.

    Distinct from :func:`fmt_bytes` so log lines that report ``len(str)`` don't
    falsely advertise byte units (``KB`` would imply UTF-8 bytes, which equals
    chars only for pure ASCII).
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{(n + 500) // 1000}K"
    return str(n)


def truncate_oneline(text: str, max_len: int = 80) -> str:
    """Collapse whitespace and clamp to ``max_len`` for single-line console output.

    Used by per-round stage markers in the codegen loops (Phase 2 dataset,
    Phase 3 reward, Phase 4 loader, Phase 4 task) to render a one-line
    rejection reason without dumping pytest output to the terminal.
    """
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 3] + "..."


def format_paper_refs(task_dirs: list[str], *, max_papers: int = 4) -> str:
    """Render a list of ``<scope>/<paper_id>`` task_dirs grouped by scope.

    Used by the Phase 2/3/4 per-candidate header lines to show which
    papers contribute to a candidate without repeating the scope name.
    Within a scope, multiple papers are grouped with brace expansion
    syntax (``scope/{paperA,paperB,...}``). Single-paper scopes render
    plain (``scope/paperA``).

    Overflow rule: when total papers across all scopes exceed
    ``max_papers``, the budget is consumed scope-by-scope in input
    order. The first scope that doesn't fit folds its surplus inside
    its own braces (``scope/{shown,+K more}``). Any scopes that appear
    after that — and could not be displayed at all — contribute to a
    trailing ``+N more`` at the end of the line. The two ``+N more``
    indicators may both appear in the same line; they are independent
    counts (within-scope overflow vs. fully-dropped-scope overflow)
    and the reader sums them mentally if total truncation matters.

    Empty ``task_dirs`` returns the empty string; callers should treat
    that as "omit the field" rather than printing ``papers=``. Entries
    without a ``/`` are emitted verbatim under an implicit empty-scope
    group (defensive; current data always has scope/paper paths).
    """
    if not task_dirs:
        return ""
    if max_papers < 1:
        max_papers = 1
    # Group by scope, preserving order of first occurrence both for the
    # scopes (outer dict insertion order) and for the papers within each
    # scope (list append order). Dedupe within a scope while we're at it.
    by_scope: dict[str, list[str]] = {}
    for raw in task_dirs:
        text = str(raw or "").strip()
        if not text:
            continue
        if "/" in text:
            scope, paper = text.split("/", 1)
        else:
            scope, paper = "", text
        bucket = by_scope.setdefault(scope, [])
        if paper not in bucket:
            bucket.append(paper)
    if not by_scope:
        return ""
    parts: list[str] = []
    remaining_budget = max_papers
    dropped_scope_papers = 0  # accumulated count of fully-dropped scopes
    for scope, papers in by_scope.items():
        if remaining_budget <= 0:
            dropped_scope_papers += len(papers)
            continue
        if len(papers) <= remaining_budget:
            parts.append(_render_scope_group(scope, papers))
            remaining_budget -= len(papers)
            continue
        # This scope overflows the remaining budget. Fold the surplus
        # into the brace group so it stays a single logical unit.
        kept = papers[:remaining_budget]
        within_surplus = len(papers) - len(kept)
        prefix = f"{scope}/" if scope else ""
        body = ",".join(kept)
        parts.append(f"{prefix}{{{body},+{within_surplus} more}}")
        remaining_budget = 0
    rendered = ",".join(parts)
    if dropped_scope_papers > 0:
        rendered += f",+{dropped_scope_papers} more"
    return rendered


def _render_scope_group(scope: str, papers: list[str]) -> str:
    prefix = f"{scope}/" if scope else ""
    if len(papers) == 1:
        return f"{prefix}{papers[0]}"
    return f"{prefix}{{{','.join(papers)}}}"


def parse_utc_date(value: str | None) -> datetime | None:
    """Parse YYYY-MM-DD into UTC datetime."""
    if not value:
        return None
    dt = datetime.strptime(value, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)
