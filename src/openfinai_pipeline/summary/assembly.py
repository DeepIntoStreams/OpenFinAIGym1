import logging

from openfinai_pipeline.summary.chunks import SummaryChunk, tokenize
from openfinai_pipeline.summary.retrieval import retrieval_intents
from openfinai_pipeline.utils.logging import log_detail

logger = logging.getLogger(__name__)
_FRONT_MATTER_CHAR_BUDGET = 4000


def _format_chunk_label(chunk_id: int, chunk: SummaryChunk) -> str:
    section = (chunk.section_title or "no-section")[:24]
    return f"{chunk_id}:p{chunk.page_number}:{section}"


def _format_top(selected: list[tuple[int, SummaryChunk]], k: int = 3) -> str:
    if not selected:
        return "(none)"
    return ", ".join(_format_chunk_label(cid, c) for cid, c in selected[:k])


def select_summary_chunks(
    chunks: list[SummaryChunk],
    ranked: dict[int, dict[str, float]],
    *,
    max_chars: int,
) -> list[SummaryChunk]:
    selected: list[tuple[int, SummaryChunk]] = []
    selected_ids: set[int] = set()
    used_chars = 0

    front_candidates = sorted(
        (
            (chunk_id, chunk)
            for chunk_id, chunk in enumerate(chunks)
            if chunk.page_number <= 2 and ranked[chunk_id]["front_matter"] > 0
        ),
        key=lambda item: (-ranked[item[0]]["front_matter"], item[1].page_number),
    )
    for chunk_id, chunk in front_candidates:
        used_chars = _try_add_chunk(
            selected=selected,
            selected_ids=selected_ids,
            used_chars=used_chars,
            chunk_id=chunk_id,
            chunk=chunk,
            max_chars=min(max_chars, _FRONT_MATTER_CHAR_BUDGET),
        )

    for intent in retrieval_intents():
        candidates = sorted(
            (
                (chunk_id, chunk)
                for chunk_id, chunk in enumerate(chunks)
                if ranked[chunk_id][intent.name] > 0
            ),
            key=lambda item: (-ranked[item[0]][intent.name], item[1].page_number),
        )
        added = 0
        for chunk_id, chunk in candidates:
            if added >= intent.max_chunks:
                break
            new_used_chars = _try_add_chunk(
                selected=selected,
                selected_ids=selected_ids,
                used_chars=used_chars,
                chunk_id=chunk_id,
                chunk=chunk,
                max_chars=max_chars,
            )
            if new_used_chars != used_chars:
                used_chars = new_used_chars
                added += 1

    remaining = sorted(
        ((chunk_id, chunk) for chunk_id, chunk in enumerate(chunks)),
        key=lambda item: (-ranked[item[0]]["overall"], item[1].page_number),
    )
    for chunk_id, chunk in remaining:
        new_used_chars = _try_add_chunk(
            selected=selected,
            selected_ids=selected_ids,
            used_chars=used_chars,
            chunk_id=chunk_id,
            chunk=chunk,
            max_chars=max_chars,
        )
        if new_used_chars == used_chars:
            continue
        used_chars = new_used_chars
        if used_chars >= max_chars:
            break

    selected.sort(key=lambda item: (item[1].page_number, item[1].section_title.lower(), item[0]))
    log_detail(
        logger,
        "summary assembly selected=%d used_chars_est=%d top=%s",
        len(selected),
        used_chars,
        _format_top(selected, k=3),
    )
    log_detail(
        logger,
        "summary assembly chunks=%s",
        ", ".join(_format_chunk_label(cid, c) for cid, c in selected) or "(none)",
    )
    return [chunk for _, chunk in selected]


def assemble_summary_excerpt(chunks: list[SummaryChunk], *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    blocks: list[str] = []
    used_chars = 0
    for chunk in chunks:
        label = f"[Page {chunk.page_number}"
        if chunk.section_title:
            label += f" | Section: {chunk.section_title}"
        label += "]"
        block = f"{label}\n{chunk.text}"
        additional = len(block) + (2 if blocks else 0)
        if blocks and used_chars + additional > max_chars:
            break
        if not blocks and len(block) > max_chars:
            log_detail(logger, "summary assembly single_chunk_truncate page=%d max_chars=%d", chunk.page_number, max_chars)
            return block[:max_chars]
        blocks.append(block)
        used_chars += additional
    excerpt = "\n\n".join(blocks).strip()
    log_detail(logger, "summary assembly excerpt_chars=%d max_chars=%d", len(excerpt), max_chars)
    return excerpt


def _try_add_chunk(
    *,
    selected: list[tuple[int, SummaryChunk]],
    selected_ids: set[int],
    used_chars: int,
    chunk_id: int,
    chunk: SummaryChunk,
    max_chars: int,
) -> int:
    if chunk_id in selected_ids or max_chars <= 0:
        return used_chars
    estimated = len(chunk.text) + 64
    if selected and used_chars + estimated > max_chars:
        return used_chars
    if any(_chunks_overlap(chunk.text, existing_chunk.text) for _, existing_chunk in selected):
        return used_chars
    selected.append((chunk_id, chunk))
    selected_ids.add(chunk_id)
    return used_chars + estimated + (2 if len(selected) > 1 else 0)


def _chunks_overlap(left: str, right: str) -> bool:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens)
    return overlap / max(min(len(left_tokens), len(right_tokens)), 1) >= 0.8
