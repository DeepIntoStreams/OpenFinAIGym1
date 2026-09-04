import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openfinai_pipeline.corpus.catalog import dataset_directory_name
from openfinai_pipeline.papers.summary_payload import extract_summary_payload

logger = logging.getLogger(__name__)
DATASET_TAG = "[phase2:dataset]"
REWARD_TAG = "[phase3:rewards]"

# Numeric-index sort so paper_2 < paper_10; unparseable names sort to end.
_PAPER_INDEX_RE = re.compile(r"^paper[_-]?(\d+)$")


def paper_index_sort_key(paper_dir_name: str) -> tuple[int, int, str]:
    """Sort key for a paper directory name.

    Returns ``(parseable_flag, numeric_index, name)``. Parseable names get
    flag 0 and sort by numeric index; unparseable names get flag 1 and sort
    by name as a fallback.
    """
    match = _PAPER_INDEX_RE.match(paper_dir_name)
    if match is not None:
        return (0, int(match.group(1)), paper_dir_name)
    return (1, sys.maxsize, paper_dir_name)


def paper_global_sort_key(task_dir: str) -> tuple[int, int, str, str]:
    """Global paper-number sort key for a ``"<scope>/<paper_dir>"`` string.

    Sort priority is: parseable-first, then numeric paper index, then scope
    name (so ``paper_1`` from every scope precedes ``paper_2`` from any
    scope), then paper directory name as a final tiebreaker. Empty/malformed
    inputs sort to the end.
    """
    parts = task_dir.split("/", 1)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return (1, sys.maxsize, "", task_dir)
    scope, paper_name = parts[0], parts[1]
    flag, idx, name = paper_index_sort_key(paper_name)
    return (flag, idx, scope, name)


@dataclass
class DatasetCandidate:
    scope_id: str
    name: str
    frequency: int
    aliases: list[str]
    description: str
    dataset_kind: str
    task_dirs: list[str]
    supporting_scope_ids: list[str]
    reproducibility_context: str = ""
    # Hint for the post-download labeler; empty when summary lacks task_family.
    task_family: str = ""
    paper_links: list[str] = field(default_factory=list)

    @property
    def directory_name(self) -> str:
        return dataset_directory_name(self.name)


@dataclass
class RewardCandidate:
    scope_id: str
    name: str
    frequency: int
    aliases: list[str]
    description: str
    task_dirs: list[str]
    supporting_scope_ids: list[str]

    @property
    def directory_name(self) -> str:
        return self.name


@dataclass
class DatasetAggregationResult:
    candidates: list[DatasetCandidate]
    raw_mentions_count: int
    # Datasets are NOT name-clustered here; reuse is decided later per-candidate
    # in `can_reuse_existing_dataset`, so this equals raw_mentions_count.
    candidate_count: int


@dataclass
class RewardAggregationResult:
    candidates: list[RewardCandidate]
    raw_mentions_count: int
    # Rewards ARE clustered by exact name (see `_dedupe_mentions_by_name`),
    # so this is the count of unique reward names.
    candidate_count: int


def load_task_summaries(research_root: Path) -> list[tuple[str, dict[str, Any]]]:
    summaries: list[tuple[str, dict[str, Any]]] = []
    if not research_root.exists():
        return summaries
    # Global paper-number ordering: every scope's paper_1 before any paper_2,
    # so downstream phases prioritise the same paper's work together.
    scope_paper_pairs: list[tuple[Path, Path]] = []
    for scope_dir in [p for p in research_root.iterdir() if p.is_dir()]:
        for paper_dir in [
            p
            for p in scope_dir.iterdir()
            if p.is_dir() and p.name.startswith("paper")
        ]:
            scope_paper_pairs.append((scope_dir, paper_dir))
    scope_paper_pairs.sort(
        key=lambda pair: paper_global_sort_key(f"{pair[0].name}/{pair[1].name}")
    )
    for scope_dir, paper_dir in scope_paper_pairs:
        paper_path = paper_dir / "paper.json"
        if not paper_path.exists():
            continue
        paper_json = json.loads(paper_path.read_text(encoding="utf-8"))
        summary = extract_summary_payload(paper_json)
        if not isinstance(summary, dict):
            continue
        scope_id = str(paper_json.get("scope_id", "")).strip() or scope_dir.name
        summary["_scope_id"] = scope_id or "default"
        summaries.append((f"{scope_dir.name}/{paper_dir.name}", summary))
    return summaries


def aggregate_datasets(
    task_summaries: list[tuple[str, dict[str, Any]]],
    existing_catalog: list[dict[str, Any]],
    llm: Any | None,
    debug_dir: Path | None = None,
) -> DatasetAggregationResult:
    del existing_catalog, llm
    mentions = _collect_dataset_mentions(task_summaries)
    logger.info(
        "%s extracted raw dataset mentions from paper summaries count=%d",
        DATASET_TAG,
        len(mentions),
    )
    candidates = _dataset_candidates_from_mentions(mentions)
    logger.info(
        "%s prepared dataset candidates without pre-reuse name clustering count=%d",
        DATASET_TAG,
        len(candidates),
    )
    # Order by global paper number; each candidate has a single task_dir at
    # this stage (no name-clustering), so we key on task_dirs[0].
    return DatasetAggregationResult(
        candidates=sorted(candidates, key=_dataset_candidate_sort_key),
        raw_mentions_count=len(mentions),
        candidate_count=len(candidates),
    )


def aggregate_rewards(
    task_summaries: list[tuple[str, dict[str, Any]]],
    existing_catalog: list[dict[str, Any]],
    llm: Any | None,
    debug_dir: Path | None = None,
) -> RewardAggregationResult:
    del existing_catalog, llm, debug_dir
    mentions = _collect_reward_mentions(task_summaries)
    logger.info(
        "%s extracted raw reward mentions from paper summaries count=%d",
        REWARD_TAG,
        len(mentions),
    )
    candidates = _reward_candidates_from_mentions(mentions)
    logger.info(
        "%s clustered raw reward mentions into unique-name candidates count=%d",
        REWARD_TAG,
        len(candidates),
    )
    # Order by earliest supporting paper so paper_1 rewards generate
    # before paper_2's (paper alignment > frequency).
    return RewardAggregationResult(
        candidates=sorted(candidates, key=_reward_candidate_sort_key),
        raw_mentions_count=len(mentions),
        candidate_count=len(candidates),
    )


def can_reuse_existing_dataset(
    candidate: DatasetCandidate,
    catalog: list[dict[str, Any]],
    llm: Any | None,
) -> tuple[bool, dict[str, Any] | None, str]:
    from openfinai_pipeline.prompts.corpus_builder import (
        build_dataset_reuse_prompt,
        dataset_reuse_schema,
    )

    relevant_catalog = [item for item in catalog if str(item.get("name", "")).strip()]
    if not relevant_catalog:
        return False, None, "no_existing_catalog"

    exact = next(
        (
            item
            for item in relevant_catalog
            if str(item.get("name", "")).strip() == candidate.name
        ),
        None,
    )
    if exact:
        return True, exact, "exact_name_match"

    if llm is None:
        return False, None, "no_llm"

    shortlist = [
        {
            "name": str(item.get("name", "")).strip(),
            "description": str(item.get("description", "")).strip()[:1000],
            "downloaded_dataset_description": str(
                item.get("downloaded_dataset_description", "")
            ).strip()[:1000],
            "preview": item.get("preview", []),
        }
        for item in relevant_catalog
    ]
    payload = llm.complete_or_fallback(
        build_dataset_reuse_prompt(
            candidate_name=candidate.name,
            candidate_aliases=[],
            candidate_description=candidate.description,
            existing_datasets=shortlist,
            candidate_reproducibility_context=candidate.reproducibility_context,
        ),
        dataset_reuse_schema(),
        fallback={"reuse": False, "matched_name": "", "reason": "fallback_no_reuse"},
    )
    reuse = bool(payload.get("reuse", False))
    matched_name = str(payload.get("matched_name", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    if not reuse or not matched_name:
        return False, None, reason or "llm_no_reuse"
    matched_entry = next(
        (
            item
            for item in relevant_catalog
            if str(item.get("name", "")).strip() == matched_name
        ),
        None,
    )
    if matched_entry is None:
        return False, None, "llm_matched_name_not_found"
    return True, matched_entry, reason or "llm_reuse"


def can_reuse_existing_reward(
    candidate: RewardCandidate,
    catalog: list[dict[str, Any]],
    llm: Any | None,
) -> tuple[bool, dict[str, Any] | None, str]:
    from openfinai_pipeline.prompts.corpus_builder import (
        build_reward_reuse_prompt,
        reward_reuse_schema,
    )

    relevant_catalog = [item for item in catalog if str(item.get("name", "")).strip()]
    if not relevant_catalog:
        return False, None, "no_existing_catalog"

    exact = next(
        (
            item
            for item in relevant_catalog
            if str(item.get("name", "")).strip() == candidate.name
        ),
        None,
    )
    if exact:
        return True, exact, "exact_name_match"
    if llm is None:
        return False, None, "no_llm"

    shortlist = [
        {
            "name": str(item.get("name", "")).strip(),
            "description": str(item.get("description", "")).strip()[:1000],
        }
        for item in relevant_catalog
    ]
    payload = llm.complete_or_fallback(
        build_reward_reuse_prompt(
            candidate_name=candidate.name,
            candidate_aliases=[],
            candidate_description=candidate.description,
            existing_rewards=shortlist,
        ),
        reward_reuse_schema(),
        fallback={"reuse": False, "matched_name": "", "reason": "fallback_no_reuse"},
    )
    reuse = bool(payload.get("reuse", False))
    matched_name = str(payload.get("matched_name", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    if not reuse or not matched_name:
        return False, None, reason or "llm_no_reuse"
    matched_entry = next(
        (
            item
            for item in relevant_catalog
            if str(item.get("name", "")).strip() == matched_name
        ),
        None,
    )
    if matched_entry is None:
        return False, None, "llm_matched_name_not_found"
    return True, matched_entry, reason or "llm_reuse"


def _collect_dataset_mentions(
    task_summaries: list[tuple[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task_dir, summary in task_summaries:
        scope_id = str(summary.get("_scope_id", "default")).strip() or "default"
        task_family = str(summary.get("task_family", "")).strip()
        paper_links = _normalize_summary_links(summary.get("links", []))
        for item in summary.get("datasets", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or name.isspace():
                logger.warning(
                    "%s skipping dataset with blank name in %s",
                    DATASET_TAG,
                    task_dir,
                )
                continue
            if len(name) > 120:
                logger.warning(
                    "%s skipping dataset with excessively long name (%d chars) in %s: %.60s...",
                    DATASET_TAG,
                    len(name),
                    task_dir,
                    name,
                )
                continue
            out.append(
                {
                    "name": name,
                    "description": _dataset_description_from_summary_item(item),
                    "dataset_kind": _normalize_dataset_kind(
                        item.get("dataset_kind", "real"),
                        task_dir=task_dir,
                        dataset_name=name,
                    ),
                    "reproducibility_context": str(
                        summary.get("experiments", "")
                    ).strip(),
                    "paper_links": paper_links,
                    "task": task_dir,
                    "scope_id": scope_id,
                    "task_family": task_family,
                }
            )
    return out


def _collect_reward_mentions(
    task_summaries: list[tuple[str, dict[str, Any]]]
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for task_dir, summary in task_summaries:
        scope_id = str(summary.get("_scope_id", "default")).strip() or "default"
        for item in summary.get("metrics", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or name.isspace():
                logger.warning(
                    "%s skipping metric with blank name in %s",
                    REWARD_TAG,
                    task_dir,
                )
                continue
            if len(name) > 80:
                logger.warning(
                    "%s skipping metric with excessively long name (%d chars) in %s: %.40s...",
                    REWARD_TAG,
                    len(name),
                    task_dir,
                    name,
                )
                continue
            out.append(
                {
                    "name": name,
                    "description": str(item.get("description", "")).strip(),
                    "task": task_dir,
                    "scope_id": scope_id,
                }
            )
    return out


def _dataset_candidate_sort_key(candidate: DatasetCandidate) -> tuple:
    """Sort datasets by global paper number, then scope, then name.

    Each dataset candidate has exactly one ``task_dir`` since dataset
    mentions are not name-clustered.
    """
    first_task_dir = candidate.task_dirs[0] if candidate.task_dirs else ""
    return paper_global_sort_key(first_task_dir) + (candidate.name,)


def _reward_candidate_sort_key(candidate: RewardCandidate) -> tuple:
    """Sort rewards by their earliest paper, then scope, then name.

    Rewards are name-clustered, so a single reward can appear in many
    papers. Using the minimum sort key across ``task_dirs`` lines a reward
    up with the first paper that introduces it.
    """
    if not candidate.task_dirs:
        return paper_global_sort_key("") + (candidate.name,)
    earliest = min(paper_global_sort_key(td) for td in candidate.task_dirs)
    return earliest + (candidate.name,)


def _dataset_candidates_from_mentions(
    mentions: list[dict[str, Any]]
) -> list[DatasetCandidate]:
    out: list[DatasetCandidate] = []
    for mention in mentions:
        name = str(mention.get("name", "")).strip()
        if not name:
            continue
        scope_id = str(mention.get("scope_id", "default")).strip() or "default"
        task_dir = str(mention.get("task", "")).strip()
        out.append(
            DatasetCandidate(
                scope_id=scope_id,
                name=name,
                frequency=1,
                aliases=[],
                description=str(mention.get("description", "")).strip(),
                dataset_kind=_normalize_dataset_kind(
                    mention.get("dataset_kind", "real"),
                    task_dir=task_dir,
                    dataset_name=name,
                ),
                task_dirs=[task_dir] if task_dir else [],
                supporting_scope_ids=[scope_id],
                reproducibility_context=str(
                    mention.get("reproducibility_context", "")
                ).strip(),
                paper_links=_normalize_summary_links(mention.get("paper_links", [])),
                task_family=str(mention.get("task_family", "")).strip(),
            )
        )
    return out


def _reward_candidates_from_mentions(
    mentions: list[dict[str, str]]
) -> list[RewardCandidate]:
    buckets = _dedupe_mentions_by_name(mentions)
    out: list[RewardCandidate] = []
    for name, bucket in buckets.items():
        owner_scope = _owning_scope(bucket["scope_counts"])
        out.append(
            RewardCandidate(
                scope_id=owner_scope,
                name=name,
                frequency=int(bucket["frequency"]),
                aliases=[],
                description=_combined_description(bucket["descriptions"]),
                task_dirs=sorted(bucket["task_dirs"]),
                supporting_scope_ids=sorted(bucket["scope_counts"].keys()),
            )
        )
    return out


def _dedupe_mentions_by_name(
    mentions: list[dict[str, str]]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for mention in mentions:
        name = str(mention.get("name", "")).strip()
        if not name:
            continue
        bucket = grouped.get(name)
        if bucket is None:
            bucket = {
                "frequency": 0,
                "descriptions": [],
                "task_dirs": set(),
                "scope_counts": {},
            }
            grouped[name] = bucket
        bucket["frequency"] += 1
        description = str(mention.get("description", "")).strip()
        if description and description not in bucket["descriptions"]:
            bucket["descriptions"].append(description)
        task_dir = str(mention.get("task", "")).strip()
        if task_dir:
            bucket["task_dirs"].add(task_dir)
        scope_id = str(mention.get("scope_id", "default")).strip() or "default"
        bucket["scope_counts"][scope_id] = (
            int(bucket["scope_counts"].get(scope_id, 0)) + 1
        )
    return grouped


def _owning_scope(scope_counts: dict[str, int]) -> str:
    if not scope_counts:
        return "default"
    return sorted(scope_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _combined_description(descriptions: list[str]) -> str:
    parts = [str(value).strip() for value in descriptions if str(value).strip()]
    return "\n\n".join(parts)


def _normalize_summary_links(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        link = str(item).strip()
        if not link or link in seen:
            continue
        seen.add(link)
        out.append(link)
    return out


def _dataset_candidates_payload(
    candidates: list[DatasetCandidate],
) -> list[dict[str, Any]]:
    return [
        {
            "scope_id": c.scope_id,
            "name": c.name,
            "frequency": c.frequency,
            "description": c.description,
            "dataset_kind": c.dataset_kind,
            "task_dirs": c.task_dirs,
            "supporting_scope_ids": c.supporting_scope_ids,
            "reproducibility_context": c.reproducibility_context,
            "paper_links": c.paper_links,
        }
        for c in candidates
    ]


def _reward_candidates_payload(
    candidates: list[RewardCandidate],
) -> list[dict[str, Any]]:
    return [
        {
            "scope_id": c.scope_id,
            "name": c.name,
            "frequency": c.frequency,
            "description": c.description,
            "task_dirs": c.task_dirs,
            "supporting_scope_ids": c.supporting_scope_ids,
        }
        for c in candidates
    ]


def _normalize_dataset_kind(
    value: Any,
    *,
    task_dir: str = "",
    dataset_name: str = "",
) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"real", "synthetic"}:
        return normalized
    context = []
    if task_dir:
        context.append(f"task={task_dir}")
    if dataset_name:
        context.append(f"dataset={dataset_name}")
    suffix = f" ({', '.join(context)})" if context else ""
    raise ValueError(
        f"Invalid dataset_kind {value!r} for dataset entry{suffix}. "
        "Expected exactly 'real' or 'synthetic'."
    )


def _dataset_description_from_summary_item(item: dict[str, Any]) -> str:
    description = str(item.get("description", "")).strip()
    generation_details = str(item.get("generation_details", "")).strip()
    if not generation_details:
        return description
    if generation_details in description:
        return description
    if not description:
        return generation_details
    return f"{description}\n\nSynthetic generation details: {generation_details}"


def _write_debug_json(debug_dir: Path | None, filename: str, payload: Any) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / filename).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
