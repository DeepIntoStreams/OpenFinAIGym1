"""Phase 1c: trading-paper triage.

For papers the judge classified as ``task_family in TRADING_FAMILIES``, this
module makes one LLM call to:

1. Pick the closest curated trading task from the catalog (currently
   ``offline_crypto_trading`` and ``offline_stock_trading``).
2. Extract per-paper task_config (symbols, target_symbols, context_resolutions,
   data_resolution, start, end, initial_cash, episode_length, slippage_pct,
   transaction_cost_pct).

Output is a ``TradingTriageRecord`` (persisted as ``trading_triage.json`` next
to ``paper.json``) plus a thin overlay under ``tasks/routed/<task_id>/`` that
re-exports the curated class with the paper-specific config baked into its
``task.toml``. Phase 2/3/4 skip routed papers; the overlay is immediately
runnable.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from openfinai_pipeline.llm import LLMService
from openfinai_pipeline.papers._curated_catalog import (
    CuratedTradingDescriptor,
    find_descriptor,
)
from openfinai_pipeline.papers.pdf_downloader import PDFDownloader
from openfinai_pipeline.papers.schemas import (
    ConfigProvenance,
    PaperRecord,
    TradingTriageAlternate,
    TradingTriageRecord,
    TradingTriageStatus,
    triage_fallback_payload,
    triage_output_schema,
)

logger = logging.getLogger(__name__)
PHASE_TAG = "[phase1:triage]"

# task_family values that route through Phase 1c instead of Phase 2-4.
TRADING_FAMILIES: frozenset[str] = frozenset(
    {"trading", "realtime_trading", "realtime_forecasting"}
)

DEFAULT_PDF_TEXT_MAX_CHARS = 200_000

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    trim_blocks=False,
    lstrip_blocks=False,
)


def is_trading_family(task_family: str | None) -> bool:
    """True if ``task_family`` should route through Phase 1c triage.

    Used at every Phase 2/3/4 filter site so the trading-family set has a
    single source of truth.
    """
    if not task_family:
        return False
    return str(task_family).strip().lower() in TRADING_FAMILIES


def generate_routed_task_id(
    curated_base: str,
    paper_id: str,
    task_name: str | None = None,
) -> str:
    """Deterministic ``routed_<curated_base>_<slug>`` task id.

    Slug source preference:
    1. Phase 1b's ``task_name`` when supplied (more readable, e.g.
       ``USEquities_DailyClose_Trading`` → ``usequities_dailyclose``).
    2. Otherwise the paper_id arxiv slug (legacy fallback).

    Both paths lowercase, replace non-alphanumerics with ``_``, and drop a
    redundant trailing ``_trading`` token (already implied by ``curated_base``).
    """
    if task_name:
        slug = re.sub(r"[^a-z0-9]+", "_", task_name.strip().lower()).strip("_")
        # curated_base already ends in "_trading"; drop redundant trailing token.
        slug = re.sub(r"(?:^|_)trading$", "", slug)
        if slug:
            return f"routed_{curated_base}_{slug}"
    raw = re.sub(r"v\d+$", "", paper_id.strip().lower())
    slug = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    if not slug:
        raise ValueError(f"paper_id {paper_id!r} produced an empty slug")
    return f"routed_{curated_base}_{slug}"


def extract_full_pdf_text(
    downloader: PDFDownloader,
    source_pdf: Path,
    *,
    max_chars: int = DEFAULT_PDF_TEXT_MAX_CHARS,
) -> tuple[str | None, bool]:
    """Return (text, was_truncated) — concatenated full PDF text or None on failure."""
    pages = downloader.extract_page_texts_from_path(source_pdf)
    if not pages:
        return None, False
    full = "\n\n".join(pages).strip()
    if not full:
        return None, False
    if len(full) <= max_chars:
        return full, False
    return full[:max_chars], True


def _build_triage_prompt(
    *,
    paper_id: str,
    pdf_text: str,
    pdf_truncated: bool,
    catalog_serialized: str,
) -> str:
    schema_json = json.dumps(triage_output_schema(), indent=2, ensure_ascii=False)
    truncation_note = (
        "\n\n(Note: the PDF text was truncated to fit context budget. The "
        "abstract, introduction, and early experimental sections are present; "
        "very late appendices may be missing.)"
        if pdf_truncated
        else ""
    )
    return f"""\
You are routing a research paper to one of a small set of curated trading
benchmarks. The paper has been classified as a trading paper by an upstream
judge; your job is NOT to re-classify, but to pick the closest curated task and
fill in the per-paper config that lets that task reproduce the paper's
backtested setup.

## Available curated trading tasks

{catalog_serialized}

## What to extract from the paper

Fill in `task_config` with every field the paper provides evidence for — this
is the primary job here, not an optional extra. The catalog defaults shown
above are a SAFETY NET for fields the paper is genuinely silent on (e.g.
`initial_cash`, `slippage_pct`); they are almost certainly wrong on the
paper-specific fields. If the paper states or strongly implies the asset
universe, backtest start/end dates, bar resolution, or episode length, you
MUST override the default by including those fields here with `explicit` or
`inferred` provenance. Leaving a paper-stated field out so the default fills
in is a routing error.

For each field you fill, record an entry in `config_provenance`:

- `"explicit"` — the paper directly states this value (quote the supporting text)
- `"inferred"` — the paper implies this value but does not state it directly
  (explain your inference)

Do NOT include `"catalog_default"` entries — those are filled in for you.

## Status rules

- `routed`: a curated task fits cleanly and the paper's setup can be reproduced
  using its config schema.
- `partial_match`: the closest curated task is the best available choice but
  doesn't fully cover the paper's setup (e.g. paper uses a derivative the
  catalog doesn't support, but the underlying spot strategy is reproducible).
  Set `routed_from` to the closest task and explain the gap in
  `gap_description`.
- `no_match`: no curated task is appropriate for this paper. Leave
  `routed_from` null and explain in `gap_description`.
- `novel_task_required`: the paper proposes a structurally novel task family
  (market-making, options spreads, multi-leg portfolio rebalancing) the catalog
  cannot represent. Leave `routed_from` null, explain in `gap_description`,
  and list the missing structural features in `novel_features`.

**Gap rule.** Only count something as a gap when the curated task's
observation/action interface or its data sources cannot represent it.
Strategy choices the agent can express on top of the existing interface are
NOT gaps — derived features computed from existing observations, self-imposed
constraints that narrow an already-broader action space (e.g. long-only when
shorts are supported; fixed size when continuous sizing is supported), and
agent-computed allocation/sizing rules (Kelly, equal-weight, market-neutral,
signal-driven stops expressible via supported order types).

## Output

Return ONLY a JSON object matching this schema (no commentary, no markdown
fences):

```json
{schema_json}
```

## Paper id

`{paper_id}`

## Paper full text{truncation_note}

{pdf_text}
"""


class TriageValidationIssue(Exception):
    """Raised when the LLM payload cannot be reconciled with the catalog."""


def _normalize_status(value: Any) -> TradingTriageStatus:
    if isinstance(value, TradingTriageStatus):
        return value
    try:
        return TradingTriageStatus(str(value).strip().lower())
    except ValueError:
        return TradingTriageStatus.NO_MATCH


def _coerce_provenance(
    raw: Any,
) -> dict[str, ConfigProvenance]:
    """Coerce LLM-provided provenance dict into pydantic models, ignoring junk."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, ConfigProvenance] = {}
    for field_name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        try:
            out[str(field_name)] = ConfigProvenance.model_validate(entry)
        except Exception:
            continue
    return out


def _coerce_alternates(raw: Any) -> list[TradingTriageAlternate]:
    if not isinstance(raw, list):
        return []
    out: list[TradingTriageAlternate] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(TradingTriageAlternate.model_validate(entry))
        except Exception:
            continue
    return out


def _merge_with_defaults(
    descriptor: CuratedTradingDescriptor,
    llm_task_config: dict[str, Any],
    llm_provenance: dict[str, ConfigProvenance],
) -> tuple[dict[str, Any], dict[str, ConfigProvenance], list[str]]:
    """Merge LLM-supplied task_config with descriptor catalog_defaults.

    Returns (merged_config, merged_provenance, missing_required_fields).
    For every field present in catalog_defaults but not in the LLM config,
    stamps a ``catalog_default`` provenance entry. Required fields with
    neither LLM value nor default are reported as missing.
    """
    merged: dict[str, Any] = dict(descriptor.catalog_defaults)
    merged.update(llm_task_config)
    provenance = dict(llm_provenance)
    for field_name in descriptor.catalog_defaults:
        if field_name not in llm_task_config and field_name not in provenance:
            provenance[field_name] = ConfigProvenance(
                source="catalog_default", evidence=None
            )
    missing_required: list[str] = []
    for field_name, spec in descriptor.config_schema.items():
        if not spec.get("required"):
            continue
        if field_name not in merged:
            missing_required.append(field_name)
    return merged, provenance, missing_required


def _cross_check_config(
    task_config: dict[str, Any],
) -> list[str]:
    """Return human-readable issue strings for cross-field invariants.

    Returning a non-empty list demotes the record to ``partial_match`` (or
    ``no_match`` if too severe) — it does not raise.
    """
    issues: list[str] = []
    symbols = task_config.get("symbols")
    target_symbols = task_config.get("target_symbols")
    context_resolutions = task_config.get("context_resolutions")
    data_resolution = task_config.get("data_resolution")

    if isinstance(symbols, list) and isinstance(target_symbols, list):
        sym_set = {str(s) for s in symbols}
        tgt_set = {str(s) for s in target_symbols}
        extras = tgt_set - sym_set
        if extras:
            issues.append(
                f"target_symbols ⊄ symbols (extras: {sorted(extras)})"
            )

    if isinstance(context_resolutions, list) and data_resolution is not None:
        intervals = []
        for r in context_resolutions:
            if isinstance(r, dict) and "interval" in r:
                intervals.append(str(r["interval"]))
        if intervals and str(data_resolution) not in intervals:
            issues.append(
                f"data_resolution {data_resolution!r} not in context_resolutions intervals "
                f"{intervals}"
            )

    return issues


def _validate_and_build_record(
    *,
    payload: dict[str, Any],
    catalog: list[CuratedTradingDescriptor],
    paper_id: str,
    catalog_version: str,
    catalog_hash: str,
    model_name: str,
    task_name: str | None = None,
) -> TradingTriageRecord:
    """Convert the raw LLM payload into a stamped, validated TradingTriageRecord."""
    status = _normalize_status(payload.get("status"))
    routed_from = payload.get("routed_from")
    if isinstance(routed_from, str):
        routed_from = routed_from.strip() or None
    confidence = float(payload.get("confidence") or 0.0)
    confidence = max(0.0, min(1.0, confidence))
    rationale = str(payload.get("rationale") or "").strip()
    llm_task_config_raw = payload.get("task_config") or {}
    llm_task_config = (
        dict(llm_task_config_raw) if isinstance(llm_task_config_raw, dict) else {}
    )
    llm_provenance = _coerce_provenance(payload.get("config_provenance"))
    alternates = _coerce_alternates(payload.get("alternates"))
    gap_description = payload.get("gap_description")
    if gap_description is not None:
        gap_description = str(gap_description).strip() or None
    novel_features_raw = payload.get("novel_features") or []
    novel_features = (
        [str(x) for x in novel_features_raw if x]
        if isinstance(novel_features_raw, list)
        else []
    )

    # Status-only paths: no routing, return early.
    if status in (
        TradingTriageStatus.NO_MATCH,
        TradingTriageStatus.NOVEL_TASK_REQUIRED,
    ):
        return TradingTriageRecord(
            catalog_version=catalog_version,
            catalog_hash=catalog_hash,
            paper_id=paper_id,
            model=model_name,
            status=status,
            routed_from=None,
            recommended_task_id=None,
            recommended_task_class=None,
            confidence=confidence,
            rationale=rationale or "(no rationale)",
            task_config={},
            config_provenance={},
            alternates=alternates,
            gap_description=gap_description,
            novel_features=novel_features,
        )

    # Routing paths require a valid catalog entry.
    descriptor = find_descriptor(catalog, routed_from) if routed_from else None
    if descriptor is None:
        # LLM picked a task that doesn't exist in the catalog — demote.
        return TradingTriageRecord(
            catalog_version=catalog_version,
            catalog_hash=catalog_hash,
            paper_id=paper_id,
            model=model_name,
            status=TradingTriageStatus.NO_MATCH,
            routed_from=None,
            recommended_task_id=None,
            recommended_task_class=None,
            confidence=confidence,
            rationale=rationale or "(no rationale)",
            task_config={},
            config_provenance={},
            alternates=alternates,
            gap_description=(
                f"LLM selected routed_from={routed_from!r} which is not in the "
                f"current catalog; demoted to no_match."
            ),
            novel_features=novel_features,
        )

    merged_config, merged_provenance, missing_required = _merge_with_defaults(
        descriptor, llm_task_config, llm_provenance
    )
    cross_check_issues = _cross_check_config(merged_config)

    # Build a status-aware gap_description that captures any newly-found issues.
    composed_gap_parts: list[str] = []
    if gap_description:
        composed_gap_parts.append(gap_description)
    if missing_required:
        composed_gap_parts.append(
            f"Missing required config fields after default-merge: "
            f"{', '.join(missing_required)}"
        )
    if cross_check_issues:
        composed_gap_parts.extend(cross_check_issues)
    composed_gap = "; ".join(composed_gap_parts) if composed_gap_parts else None

    final_status = status
    if missing_required:
        final_status = TradingTriageStatus.NO_MATCH
        return TradingTriageRecord(
            catalog_version=catalog_version,
            catalog_hash=catalog_hash,
            paper_id=paper_id,
            model=model_name,
            status=final_status,
            routed_from=None,
            recommended_task_id=None,
            recommended_task_class=None,
            confidence=confidence,
            rationale=rationale or "(no rationale)",
            task_config={},
            config_provenance={},
            alternates=alternates,
            gap_description=composed_gap,
            novel_features=novel_features,
        )
    if cross_check_issues and final_status == TradingTriageStatus.ROUTED:
        final_status = TradingTriageStatus.PARTIAL_MATCH

    recommended_task_id = generate_routed_task_id(
        descriptor.task_id, paper_id, task_name=task_name
    )
    return TradingTriageRecord(
        catalog_version=catalog_version,
        catalog_hash=catalog_hash,
        paper_id=paper_id,
        model=model_name,
        status=final_status,
        routed_from=descriptor.task_id,
        recommended_task_id=recommended_task_id,
        recommended_task_class=descriptor.class_name,
        confidence=confidence,
        rationale=rationale or "(no rationale)",
        task_config=merged_config,
        config_provenance=merged_provenance,
        alternates=alternates,
        gap_description=composed_gap,
        novel_features=novel_features,
    )


def triage_paper(
    *,
    paper_dir: Path,
    paper_record: PaperRecord,
    catalog: list[CuratedTradingDescriptor],
    catalog_version: str,
    catalog_hash: str,
    llm: LLMService,
    downloader: PDFDownloader,
    model_name: str,
    pdf_text_max_chars: int = DEFAULT_PDF_TEXT_MAX_CHARS,
    catalog_serialized: str | None = None,
    task_name: str | None = None,
) -> TradingTriageRecord:
    """Triage one paper. Reads its PDF, calls the LLM, validates, returns record.

    Raises:
        FileNotFoundError: if ``paper_dir / "source.pdf"`` is missing.
        TriageValidationIssue: if the PDF text cannot be extracted.
    """
    source_pdf = paper_dir / "source.pdf"
    if not source_pdf.exists():
        raise FileNotFoundError(f"missing source.pdf for paper at {paper_dir}")

    pdf_text, truncated = extract_full_pdf_text(
        downloader, source_pdf, max_chars=pdf_text_max_chars
    )
    if not pdf_text:
        raise TriageValidationIssue(
            f"could not extract PDF text for paper at {paper_dir}"
        )

    if catalog_serialized is None:
        from openfinai_pipeline.papers._curated_catalog import (
            serialize_catalog_for_prompt,
        )

        catalog_serialized = serialize_catalog_for_prompt(catalog)

    prompt = _build_triage_prompt(
        paper_id=paper_record.paper_id,
        pdf_text=pdf_text,
        pdf_truncated=truncated,
        catalog_serialized=catalog_serialized,
    )
    payload = llm.complete_or_fallback(
        prompt,
        triage_output_schema(),
        fallback=triage_fallback_payload(),
    )
    return _validate_and_build_record(
        payload=payload,
        catalog=catalog,
        paper_id=paper_record.paper_id,
        catalog_version=catalog_version,
        catalog_hash=catalog_hash,
        model_name=model_name,
        task_name=task_name,
    )


_BUNDLE_MIRROR_SUBDIRS: tuple[str, ...] = ("tests", "environment")
_BUNDLE_MIRROR_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def _mirror_base_bundle_assets(base_task_dir: Path, overlay_dir: Path) -> None:
    """Mirror ``tests/`` + ``environment/`` from the curated base into the overlay.

    Harbor's Task loader requires ``tests/test.sh`` (verifier-mode SHARED) and
    our runner bind-mounts ``environment/data`` at ``/data``; routed bundles
    inherit both wholesale from the base so they dispatch as first-class
    curated bundles. Each target subtree is removed and re-copied so
    re-materialization stays byte-equivalent to the base.
    """
    for sub in _BUNDLE_MIRROR_SUBDIRS:
        src = base_task_dir / sub
        if not src.is_dir():
            raise FileNotFoundError(
                f"base curated bundle is missing required subdir: {src}"
            )
        dst = overlay_dir / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=_BUNDLE_MIRROR_IGNORE)


def render_overlay(
    *,
    record: TradingTriageRecord,
    descriptor: CuratedTradingDescriptor,
    overlay_root: Path,
    tasks_root: Path,
    paper_pdf: Path | None = None,
) -> Path:
    """Render the per-paper routed bundle under ``overlay_root / task_id``.

    Writes ``task.toml`` (with the base curated ``family`` so harbor
    dispatches it as a first-class curated task), a ``task.py`` shim that
    re-exports the base task class, ``instruction.md``, ``__init__.py``, and
    ``triage_record.json``. Also mirrors ``tests/`` and ``environment/``
    from ``tasks_root / record.routed_from`` so the bundle is self-contained
    and runnable by Harbor's curated task loader without routed-awareness.

    When ``paper_pdf`` is supplied and the file exists, it is copied to
    ``<overlay_dir>/<task_id>.pdf`` to match the generated-task convention
    (see ``install_task`` in benchmark/builders/task_builder.py). Missing
    or unreadable PDF is logged at warning level and does NOT fail the
    render — the PDF is a human-reviewer convenience, not a runtime
    dependency.

    Idempotent — overwrites any existing files. Only valid for
    routed/partial_match records; raises for no_match/novel.
    """
    if record.status in (
        TradingTriageStatus.NO_MATCH,
        TradingTriageStatus.NOVEL_TASK_REQUIRED,
    ):
        raise ValueError(
            f"cannot render overlay for status={record.status.value} "
            f"(paper_id={record.paper_id})"
        )
    if record.recommended_task_id is None or record.routed_from is None:
        raise ValueError("record is missing recommended_task_id or routed_from")

    base_task_dir = Path(tasks_root) / record.routed_from
    if not base_task_dir.is_dir():
        raise FileNotFoundError(
            f"base curated task directory not found: {base_task_dir} "
            f"(routed_from={record.routed_from!r})"
        )

    overlay_dir = overlay_root / record.recommended_task_id
    overlay_dir.mkdir(parents=True, exist_ok=True)

    config = record.task_config
    context = {
        "task_id": record.recommended_task_id,
        "routed_from": record.routed_from,
        "class_name": record.recommended_task_class,
        "paper_id": record.paper_id,
        # Base curated identity so the materialized bundle dispatches as a
        # first-class curated task (not a routed_* family harbor can't load).
        "base_family": descriptor.base_family,
        "docker_image": descriptor.docker_image,
        "task_description": descriptor.task_description,
        "action_space_description": descriptor.action_space_description,
        "action_format_description": descriptor.action_format_description,
        "evaluation_description": descriptor.evaluation_description,
        # Direct config fields exposed to both templates:
        "symbols": config.get("symbols", []),
        "target_symbols": config.get("target_symbols", []),
        "context_resolutions": config.get("context_resolutions", []),
        "data_resolution": config.get("data_resolution", ""),
        "start": config.get("start", ""),
        "end": config.get("end", ""),
        "initial_cash": config.get("initial_cash", 0.0),
        "episode_length": config.get("episode_length", 0),
        "start_offset": config.get("start_offset", 0),
        "slippage_pct": config.get("slippage_pct", 0.0),
        "transaction_cost_pct": config.get("transaction_cost_pct", 0.0),
    }

    (overlay_dir / "task.toml").write_text(
        _jinja_env.get_template("routed_task.toml.j2").render(**context),
        encoding="utf-8",
    )
    (overlay_dir / "task.py").write_text(
        _jinja_env.get_template("routed_task.py.j2").render(**context),
        encoding="utf-8",
    )
    (overlay_dir / "instruction.md").write_text(
        _jinja_env.get_template("routed_instruction.md.j2").render(**context),
        encoding="utf-8",
    )
    (overlay_dir / "__init__.py").write_text(
        _jinja_env.get_template("routed_init.py.j2").render(**context),
        encoding="utf-8",
    )
    (overlay_dir / "triage_record.json").write_text(
        json.dumps(record.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _mirror_base_bundle_assets(base_task_dir, overlay_dir)

    if paper_pdf is not None:
        pdf_src = Path(paper_pdf)
        pdf_dst = overlay_dir / f"{record.recommended_task_id}.pdf"
        if pdf_src.is_file():
            try:
                shutil.copy2(pdf_src, pdf_dst)
            except OSError as exc:
                logger.warning(
                    "%s pdf_copy_failed task_id=%s src=%s error=%s",
                    PHASE_TAG,
                    record.recommended_task_id,
                    pdf_src,
                    exc,
                )
        else:
            logger.warning(
                "%s pdf_copy_skipped task_id=%s reason=missing src=%s",
                PHASE_TAG,
                record.recommended_task_id,
                pdf_src,
            )

    return overlay_dir
