import re
from dataclasses import dataclass

_CHUNK_TARGET_CHARS = 900
_MIN_TOKEN_LEN = 2
_TOKEN_PATTERN = re.compile(r"[a-z0-9_./:+-]+", flags=re.IGNORECASE)
_SECTION_HEADING_PATTERN = re.compile(
    r"^(?:appendix(?:\s+[a-z0-9]+)*(?:\s+[A-Za-z][A-Za-z0-9 ,:/()\-]*)?|abstract|introduction|background|"
    r"related work|related works|method|methods|model|models|algorithm|algorithms|data|dataset|datasets|"
    r"experimental setup|experiments|evaluation|implementation details|implementation|training details|"
    r"hyperparameters?|results|discussion|conclusion|supplementary materials?|references|bibliography|"
    r"acknowledg(?:e?ments?|ments?)|"
    r"(?:[a-z]|\d+(?:\.\d+)*)[\).:]?\s+[A-Z][A-Za-z0-9 ,:/()\-]{2,})$",
    flags=re.IGNORECASE,
)
_REFERENCES_HEADING_PATTERN = re.compile(
    r"^(?:\d+\.?\s*)?(references|bibliography|works\s+cited)\s*$",
    flags=re.IGNORECASE,
)
# Drop from prompt context: References/Bibliography, Related Work, and
# Acknowledgements. Pattern matches only clean standalone headings, never
# body-text substrings.
_LOW_VALUE_HEADING_PATTERN = re.compile(
    r"^(?:\d+\.?\s*)?("
    r"references|bibliography|works\s+cited|"
    r"related\s+works?|"
    r"acknowledg(?:e?ments?|ments?)"
    r")\s*$",
    flags=re.IGNORECASE,
)
# X.Y-style heading (digit/letter + dot + digit) belongs to its parent section,
# so it must NOT terminate the skip window when a parent is being stripped.
_SUB_HEADING_NUMBER_PATTERN = re.compile(r"^(?:[A-Z]|\d+)\.\d", flags=re.IGNORECASE)


def _is_top_level_heading(stripped: str) -> bool:
    """A heading is top-level unless its leading numbering looks like X.Y."""
    return not _SUB_HEADING_NUMBER_PATTERN.match(stripped)


@dataclass(frozen=True)
class SummaryChunk:
    page_number: int
    section_title: str
    text: str


def extract_summary_chunks(page_texts: list[str]) -> list[SummaryChunk]:
    chunks: list[SummaryChunk] = []
    current_section = ""
    for page_number, page_text in enumerate(page_texts, start=1):
        lines = [line.strip() for line in (page_text or "").splitlines()]
        paragraphs: list[tuple[str, str]] = []
        buffer: list[str] = []
        buffer_section = current_section
        for line in lines:
            if not line:
                if buffer:
                    paragraphs.append((buffer_section, " ".join(buffer).strip()))
                    buffer = []
                continue
            if _is_section_heading(line):
                if buffer:
                    paragraphs.append((buffer_section, " ".join(buffer).strip()))
                    buffer = []
                current_section = line
                buffer_section = line
                continue
            buffer.append(line)
        if buffer:
            paragraphs.append((buffer_section, " ".join(buffer).strip()))

        for section_title, paragraph in _merge_paragraphs(paragraphs):
            normalized = normalize_text(paragraph)
            if normalized:
                chunks.append(
                    SummaryChunk(
                        page_number=page_number,
                        section_title=section_title,
                        text=normalized,
                    )
                )
    return chunks


def normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in _TOKEN_PATTERN.findall(text or "")]
    return [token for token in tokens if len(token) >= _MIN_TOKEN_LEN]


def _merge_paragraphs(paragraphs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: list[tuple[str, str]] = []
    buffer: list[str] = []
    current_section = ""
    current_len = 0
    for section_title, paragraph in paragraphs:
        normalized = normalize_text(paragraph)
        if not normalized:
            continue
        if buffer and (
            section_title != current_section
            or current_len + len(normalized) > _CHUNK_TARGET_CHARS
        ):
            merged.append((current_section, " ".join(buffer).strip()))
            buffer = [normalized]
            current_section = section_title
            current_len = len(normalized)
            continue
        if not buffer:
            current_section = section_title
        buffer.append(normalized)
        current_len += len(normalized)
    if buffer:
        merged.append((current_section, " ".join(buffer).strip()))
    return merged


def _is_section_heading(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped or len(stripped) > 120:
        return False
    if stripped.endswith(".") and len(stripped.split()) > 4:
        return False
    return bool(_SECTION_HEADING_PATTERN.match(stripped))


def _strip_sections_matching(text: str, section_pattern: re.Pattern[str]) -> str:
    """Drop content under any heading matching ``section_pattern``.

    Walks line-by-line. When a matching heading is hit, subsequent lines
    are skipped until the next *top-level* (non-matching) section heading
    appears, at which point output resumes. Sub-section headings inside
    the skipped section (e.g. ``2.1 Traditional Methods`` under
    ``2 RELATED WORKS``) do NOT terminate the skip — they belong to the
    parent. This preserves later sections (e.g. appendices that follow
    References). Uses the guarded ``_is_section_heading`` to avoid
    body-line false positives.
    """
    out_lines: list[str] = []
    in_skip = False
    for line in (text or "").splitlines():
        stripped = line.strip()
        if _is_section_heading(stripped):
            if section_pattern.match(stripped):
                in_skip = True
                continue
            if in_skip and not _is_top_level_heading(stripped):
                continue
            in_skip = False
        if not in_skip:
            out_lines.append(line)
    return "\n".join(out_lines)


def strip_references_section(text: str) -> str:
    """Drop content under References/Bibliography headings, preserve any following sections."""
    return _strip_sections_matching(text, _REFERENCES_HEADING_PATTERN)


def strip_low_value_sections(text: str) -> str:
    """Drop References, Related Work, and Acknowledgements sections in one pass.

    Used by Phase 4 paper-context extraction (loader + task assembly) and the
    Phase 1 sift judge to recover prompt budget without losing main-body
    content. Sections that come after a stripped section (e.g. appendices
    after References) are preserved.
    """
    return _strip_sections_matching(text, _LOW_VALUE_HEADING_PATTERN)
