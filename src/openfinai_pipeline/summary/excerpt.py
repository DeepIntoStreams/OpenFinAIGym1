import logging

from openfinai_pipeline.settings import SummaryConfig
from openfinai_pipeline.summary.assembly import assemble_summary_excerpt, select_summary_chunks
from openfinai_pipeline.summary.chunks import extract_summary_chunks
from openfinai_pipeline.summary.retrieval import rank_summary_chunks
from openfinai_pipeline.utils.logging import fmt_chars

logger = logging.getLogger(__name__)


def build_targeted_summary_excerpt(
    page_texts: list[str],
    *,
    max_chars: int,
    retrieval_config: SummaryConfig,
) -> str:
    if max_chars <= 0:
        return ""
    chunks = extract_summary_chunks(page_texts)
    if not chunks:
        return ""
    ranked = rank_summary_chunks(chunks, config=retrieval_config)
    selected = select_summary_chunks(chunks, ranked, max_chars=max_chars)
    excerpt = assemble_summary_excerpt(selected, max_chars=max_chars)
    raw_chars = sum(len(p) for p in page_texts)
    logger.info(
        "excerpt pages=%d raw=%s chunks=%d selected=%d final=%s (cap=%s)",
        len(page_texts),
        fmt_chars(raw_chars),
        len(chunks),
        len(selected),
        fmt_chars(len(excerpt)),
        fmt_chars(max_chars),
    )
    return excerpt
