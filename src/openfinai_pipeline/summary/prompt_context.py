import logging
from pathlib import Path

from openfinai_pipeline.papers.pdf_downloader import PDFDownloader
from openfinai_pipeline.settings import SummaryConfig
from openfinai_pipeline.summary.excerpt import build_targeted_summary_excerpt
from openfinai_pipeline.utils.logging import fmt_chars, log_detail

DEFAULT_SUMMARY_EXCERPT_CHARS = 80000
logger = logging.getLogger(__name__)


def build_summary_excerpt_for_prompt(
    downloader: PDFDownloader,
    source_pdf: Path,
    summary_excerpt_chars: int,
    retrieval_config: SummaryConfig,
) -> str | None:
    page_texts = downloader.extract_page_texts_from_path(source_pdf)
    target_chars = DEFAULT_SUMMARY_EXCERPT_CHARS if summary_excerpt_chars == 0 else summary_excerpt_chars
    log_detail(
        logger,
        "summary prompt context source_pdf=%s requested_chars=%s target_chars=%s page_texts=%s",
        source_pdf,
        summary_excerpt_chars,
        target_chars,
        0 if page_texts is None else len(page_texts),
    )
    if page_texts:
        excerpt = build_targeted_summary_excerpt(
            page_texts,
            max_chars=target_chars,
            retrieval_config=retrieval_config,
        )
        if excerpt:
            log_detail(
                logger,
                "summary prompt context assembled_excerpt_chars=%d source_pdf=%s",
                len(excerpt),
                source_pdf,
            )
            return excerpt
    fallback = downloader.extract_excerpt_from_path(source_pdf, max_chars=target_chars)
    if not fallback:
        logger.warning("summary prompt context missing_excerpt source_pdf=%s", source_pdf)
        return None
    logger.info(
        "excerpt fallback final=%s (cap=%s)",
        fmt_chars(len(fallback)),
        fmt_chars(target_chars),
    )
    return fallback
