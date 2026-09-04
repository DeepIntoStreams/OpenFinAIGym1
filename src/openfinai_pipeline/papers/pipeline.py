import hashlib
import json
import logging
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openfinai_pipeline.papers.collectors.arxiv import (
    ArxivClient,
    since_days_to_datetime,
)
from openfinai_pipeline.papers.collectors.crossref import CrossrefClient
from openfinai_pipeline.papers.collectors.semantic_scholar import (
    SemanticScholarClient,
)
from openfinai_pipeline.papers._curated_catalog import (
    compute_catalog_hash,
    load_curated_trading_catalog,
    serialize_catalog_for_prompt,
)
from openfinai_pipeline.papers.judge import JudgeService
from openfinai_pipeline.papers.pdf_downloader import PDFDownloader
from openfinai_pipeline.papers.schemas import (
    JudgeDecision,
    JudgeLabel,
    PaperRecord,
    PaperStatus,
    PaperTaskBinding,
    RunSummary,
    ScopeDefinition,
    TradingTriageStatus,
)
from openfinai_pipeline.papers.storage import ResearchStore
from openfinai_pipeline.papers.summary_payload import (
    extract_summary_payload,
    has_summary_payload,
)
from openfinai_pipeline.papers.triage import (
    DEFAULT_PDF_TEXT_MAX_CHARS,
    TriageValidationIssue,
    is_trading_family,
    render_overlay,
    triage_paper,
)
from openfinai_pipeline.settings import (
    AppConfig,
    JudgeConfig,
    filter_scopes,
    load_app_config,
    load_scope_definitions,
    resolve_phase_llm,
)
from openfinai_pipeline.summary.prompt_context import (
    DEFAULT_SUMMARY_EXCERPT_CHARS,
    build_summary_excerpt_for_prompt,
)
from openfinai_pipeline.utils.artifacts import ArtifactWriter
from openfinai_pipeline.utils.logging import (
    configure_practice_logging,
    fmt_bytes,
    fmt_chars,
    log_stage,
    parse_utc_date,
    truncate_oneline,
)

logger = logging.getLogger(__name__)
PHASE_TAG = "[phase1:scrape]"
SUMMARY_TAG = "[phase1:summarize]"
TRIAGE_TAG = "[phase1:triage]"
IMPORT_TAG = "[phase1:manual-import]"


def run_scraping(
    config_path: str | None = None,
    scopes_override: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int | None = None,
    max_accepts: int | None = None,
    log_practice: str = "scrape",
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    llm_max_retries: int | None = None,
    temperature: float | None = None,
    timeout_sec: float | None = None,
    verbose: bool = False,
    overwrite: bool = False,
) -> RunSummary:
    """Execute one run across selected scopes and persist accepted-paper artifacts.

    Rerun dedupe (default behavior, ``overwrite=False``):
        After collection, papers whose ``paper_id`` already appears in
        this scope's ``index.json`` — either as an accepted entry or in
        ``seen_rejected`` — are filtered out. This skips the expensive
        downstream work (excerpt prefetch, LLM prefilter, sift, full-PDF
        download) on papers we've already judged. Per-scope partitioning
        means a paper rejected under scope A may legitimately be re-judged
        under scope B (different queries/categories/threshold).

    With ``overwrite=True``:
        Dedupe filter is bypassed; every collected paper is re-judged.
        Accepted re-judgments reuse the existing ``paperN/`` slot (no new
        directory) and overwrite ``source.pdf``. Because existing accepts
        are themselves re-judged this run, the accept-cap does NOT subtract
        them — the budget is the full ``per_scope_max_accepts`` (a fresh
        slate of up to that many accepts), not ``per_scope_max_accepts``
        minus existing.
    """
    cfg = load_app_config(config_path)
    llm_cfg = resolve_phase_llm(
        cfg.llm,
        "scrape",
        cli_provider=provider,
        cli_model=model,
        cli_max_tokens=max_tokens,
        cli_temperature=temperature,
        cli_llm_max_retries=llm_max_retries,
        cli_timeout_sec=timeout_sec,
    )

    all_scopes = load_scope_definitions(cfg.scopes.path)
    scopes = filter_scopes(all_scopes, scopes_override)

    summary = _new_summary(scopes)
    run_tag = f"{summary.run_id}_{log_practice}"
    run_dir = configure_practice_logging(
        cfg.logging.dir, log_practice, run_tag=run_tag, verbose=verbose
    )
    artifacts = ArtifactWriter(cfg.logging.dir)
    research_store = ResearchStore(cfg.papers.root_dir)

    arxiv = ArxivClient(cfg.collectors.arxiv)
    crossref = CrossrefClient(cfg.collectors.crossref)
    s2 = SemanticScholarClient(cfg.collectors.semantic_scholar)
    judge = JudgeService(cfg, llm_cfg)
    downloader = PDFDownloader(cfg.download)

    since_date = parse_utc_date(since) or since_days_to_datetime(cfg.run.since_days)
    until_date = parse_utc_date(until) or datetime.now(tz=timezone.utc)
    per_scope_limit = (
        min(limit, cfg.run.max_papers_per_scope)
        if limit
        else cfg.run.max_papers_per_scope
    )
    per_scope_max_accepts = (
        max_accepts if max_accepts is not None else cfg.run.max_accepts_per_scope
    )
    sift_budget = _effective_sift_budget(cfg.judge, per_scope_limit)

    log_stage(
        logger,
        "%s start run_id=%s scopes=%d provider=%s model=%s",
        PHASE_TAG,
        summary.run_id,
        len(scopes),
        llm_cfg.provider,
        llm_cfg.providers[llm_cfg.provider].model,
    )
    logger.info(
        "%s start detail config=%s scopes_file=%s selected_scopes=%s papers_root=%s logs_dir=%s",
        PHASE_TAG,
        config_path or "default",
        cfg.scopes.path,
        [scope.id for scope in scopes],
        cfg.papers.root_dir,
        run_dir,
    )
    logger.info(
        "%s runtime window since=%s until=%s per_scope_limit=%s per_scope_max_accepts=%s sift_budget=%d",
        PHASE_TAG,
        since_date.isoformat() if since_date else "None",
        until_date.isoformat(),
        per_scope_limit,
        per_scope_max_accepts,
        sift_budget,
    )
    logger.info(
        "%s load scopes total=%d enabled_selected=%d from=%s",
        PHASE_TAG,
        len(all_scopes),
        len(scopes),
        cfg.scopes.path,
    )

    judgment_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []

    try:
        for scope_index, scope in enumerate(scopes, start=1):
            log_stage(
                logger,
                "%s [%d/%d] scope=%s begin queries=%d categories=%d limit=%d",
                PHASE_TAG,
                scope_index,
                len(scopes),
                scope.id,
                len(scope.queries),
                len(scope.categories),
                per_scope_limit,
            )
            papers = _collect_scope_papers(
                summary=summary,
                arxiv=arxiv,
                crossref=crossref,
                s2=s2,
                scope=scope,
                since_date=since_date,
                until_date=until_date,
                per_scope_limit=per_scope_limit,
            )

            papers = _apply_seen_dedupe(
                summary=summary,
                research_store=research_store,
                scope=scope,
                papers=papers,
                overwrite=overwrite,
            )

            prefetch_excerpts = _prefetch_excerpts(
                cfg=cfg, downloader=downloader, papers=papers
            )
            decisions = _judge_scope(
                summary=summary,
                judge=judge,
                scope=scope,
                papers=papers,
                excerpts=prefetch_excerpts,
                sift_budget=sift_budget,
            )

            decisions, downloaded_pdf_paths = _download_accepted_pdfs(
                cfg=cfg,
                summary=summary,
                judge=judge,
                downloader=downloader,
                scope=scope,
                papers=papers,
                decisions=decisions,
                prefetch_excerpts=prefetch_excerpts,
                per_scope_max_accepts=per_scope_max_accepts,
                research_store=research_store,
                overwrite=overwrite,
            )

            scope_judgment_rows, scope_accepted_rows = _persist_scope_results(
                summary=summary,
                research_store=research_store,
                scope=scope,
                run_id=summary.run_id,
                papers=papers,
                decisions=decisions,
                downloaded_pdf_paths=downloaded_pdf_paths,
            )
            judgment_rows.extend(scope_judgment_rows)
            accepted_rows.extend(scope_accepted_rows)
            log_stage(
                logger,
                "%s scope=%s complete judged_candidates=%d accepted=%d rejected=%d persisted=%d pdf_downloaded=%d",
                PHASE_TAG,
                scope.id,
                len(papers),
                sum(
                    1 for decision in decisions if decision.label == JudgeLabel.ACCEPTED
                ),
                sum(
                    1 for decision in decisions if decision.label != JudgeLabel.ACCEPTED
                ),
                len(scope_accepted_rows),
                len(downloaded_pdf_paths),
            )

        summary.status = "completed"
    except Exception as e:
        summary.status = "failed"
        summary.errors.append(str(e))
        logger.exception("%s failed run_id=%s error=%s", PHASE_TAG, summary.run_id, e)
        raise
    finally:
        summary.finished_at = datetime.now(tz=timezone.utc).isoformat()
        _sort_judgment_rows(judgment_rows)
        _sort_accepted_rows(accepted_rows)
        artifacts.write_json(run_tag, "scrape_summary", summary.model_dump())
        artifacts.write_csv(run_tag, "scrape_judgments", judgment_rows)
        artifacts.write_csv(run_tag, "accepted_papers", accepted_rows)
        logger.info(
            "%s write artifacts run_dir=%s files=%s",
            PHASE_TAG,
            run_dir,
            ["scrape_summary.json", "scrape_judgments.csv", "accepted_papers.csv"],
        )
        log_stage(
            logger,
            "%s complete run_id=%s status=%s scraped=%d enriched=%d "
            "skipped_seen_accepted=%d skipped_seen_rejected=%d "
            "judged=%d accepted=%d rejected=%d downloaded=%d persisted=%d",
            PHASE_TAG,
            summary.run_id,
            summary.status,
            summary.total_scraped,
            summary.total_enriched,
            summary.total_skipped_seen_accepted,
            summary.total_skipped_seen_rejected,
            summary.total_judged,
            summary.total_accepted,
            summary.total_rejected,
            summary.total_downloaded,
            summary.total_tasks_constructed,
        )

    return summary


def summarize_papers(
    config_path: str | None = None,
    scopes_override: list[str] | None = None,
    overwrite: bool = False,
    summary_excerpt_chars: int = DEFAULT_SUMMARY_EXCERPT_CHARS,
    log_practice: str = "summarize_paper",
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    llm_max_retries: int | None = None,
    temperature: float | None = None,
    timeout_sec: float | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    if summary_excerpt_chars < 0:
        raise ValueError("summary_excerpt_chars must be >= 0")
    cfg = load_app_config(config_path)
    llm_cfg = resolve_phase_llm(
        cfg.llm,
        "summarize",
        cli_provider=provider,
        cli_model=model,
        cli_max_tokens=max_tokens,
        cli_temperature=temperature,
        cli_llm_max_retries=llm_max_retries,
        cli_timeout_sec=timeout_sec,
    )
    all_scopes = load_scope_definitions(cfg.scopes.path)
    scopes = filter_scopes(all_scopes, scopes_override)
    scope_map = {scope.id: scope for scope in scopes}

    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
    run_tag = f"{run_id}_{log_practice}"
    run_dir = configure_practice_logging(
        cfg.logging.dir, log_practice, run_tag=run_tag, verbose=verbose
    )
    artifacts = ArtifactWriter(cfg.logging.dir)
    research_store = ResearchStore(cfg.papers.root_dir)
    judge = JudgeService(cfg, llm_cfg)
    downloader = PDFDownloader(cfg.download)

    summary = {
        "run_id": run_id,
        "status": "completed",
        "summary_excerpt_chars": summary_excerpt_chars,
        "summarized": 0,
        "skipped_existing": 0,
        "skipped_missing_pdf": 0,
        "skipped_not_accepted": 0,
        "skipped_scope": 0,
        "failed": 0,
    }
    rows: list[dict[str, Any]] = []
    task_dirs = [
        p
        for p in research_store.iter_task_dirs()
        if not scope_map or p.parent.name in scope_map
    ]

    log_stage(
        logger,
        "%s start run_id=%s candidate_papers=%d provider=%s model=%s",
        SUMMARY_TAG,
        run_id,
        len(task_dirs),
        llm_cfg.provider,
        llm_cfg.providers[llm_cfg.provider].model,
    )
    logger.info(
        "%s start detail config=%s selected_scopes=%s papers_root=%s logs_dir=%s overwrite=%s summary_excerpt_chars=%s",
        SUMMARY_TAG,
        config_path or "default",
        [scope.id for scope in scopes],
        cfg.papers.root_dir,
        run_dir,
        overwrite,
        summary_excerpt_chars,
    )

    for paper_index, paper_dir in enumerate(task_dirs, start=1):
        rel_paper_dir = str(paper_dir.relative_to(research_store._root))
        scope_id = paper_dir.parent.name

        paper_json_path = paper_dir / "paper.json"
        if not paper_json_path.exists():
            summary["failed"] += 1
            rows.append({"paper_dir": rel_paper_dir, "status": "missing_paper_json"})
            continue

        paper_doc = json.loads(paper_json_path.read_text(encoding="utf-8"))
        scope_id = str(paper_doc.get("scope_id", scope_id)).strip() or scope_id
        scope = scope_map.get(scope_id)
        if scope is None:
            # paper.json overrides the directory's scope_id to something out-of-run
            summary["skipped_scope"] += 1
            continue

        paper_label = rel_paper_dir.replace("\\", "/")
        log_stage(
            logger,
            "%s [%d/%d] paper=%s",
            SUMMARY_TAG,
            paper_index,
            len(task_dirs),
            paper_label,
        )

        decision_doc = paper_doc.get("decision") or {}
        if str(decision_doc.get("label", "")).strip().lower() != "accepted":
            summary["skipped_not_accepted"] += 1
            rows.append(
                {
                    "paper_dir": rel_paper_dir,
                    "scope_id": scope_id,
                    "status": "skipped_not_accepted",
                }
            )
            log_stage(logger, "%s   -> skip (not_accepted)", SUMMARY_TAG)
            continue

        if has_summary_payload(paper_doc) and not overwrite:
            summary["skipped_existing"] += 1
            rows.append(
                {
                    "paper_dir": rel_paper_dir,
                    "scope_id": scope_id,
                    "status": "skipped_existing",
                }
            )
            log_stage(logger, "%s   -> skip (existing summary)", SUMMARY_TAG)
            continue

        source_pdf = paper_dir / "source.pdf"
        if not source_pdf.exists():
            summary["skipped_missing_pdf"] += 1
            rows.append(
                {
                    "paper_dir": rel_paper_dir,
                    "scope_id": scope_id,
                    "status": "missing_source_pdf",
                }
            )
            log_stage(logger, "%s   -> skip (missing source.pdf)", SUMMARY_TAG)
            continue

        excerpt = build_summary_excerpt_for_prompt(
            downloader=downloader,
            source_pdf=source_pdf,
            summary_excerpt_chars=summary_excerpt_chars,
            retrieval_config=cfg.summary,
        )
        if not excerpt:
            summary["skipped_missing_pdf"] += 1
            rows.append(
                {
                    "paper_dir": rel_paper_dir,
                    "scope_id": scope_id,
                    "status": "missing_pdf_excerpt",
                }
            )
            log_stage(logger, "%s   -> skip (no excerpt)", SUMMARY_TAG)
            continue

        try:
            paper = PaperRecord.model_validate(paper_doc.get("paper") or {})
            payload = judge.summarize_paper(scope, paper, excerpt)
            research_store.persist_paper_summary(
                paper_dir=paper_dir,
                summary_payload=payload,
            )
            summary["summarized"] += 1
            dataset_count = len(payload.get("datasets") or [])
            metric_count = len(payload.get("metrics") or [])
            task_family = payload.get("task_family") or "?"
            rows.append(
                {
                    "paper_dir": rel_paper_dir,
                    "scope_id": scope_id,
                    "status": "summarized",
                    "summary_excerpt_chars": len(excerpt),
                    "task_family": task_family,
                    "dataset_count": dataset_count,
                    "metric_count": metric_count,
                }
            )
            log_stage(
                logger,
                "%s   -> done task_family=%s datasets=%d metrics=%d",
                SUMMARY_TAG,
                task_family,
                dataset_count,
                metric_count,
            )
        except Exception as exc:
            summary["failed"] += 1
            rows.append(
                {
                    "paper_dir": rel_paper_dir,
                    "scope_id": scope_id,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            log_stage(
                logger,
                "%s   -> failed: %s",
                SUMMARY_TAG,
                truncate_oneline(str(exc)),
            )
            logger.exception(
                "%s failed paper_dir=%s error=%s", SUMMARY_TAG, rel_paper_dir, exc
            )

    if summary["failed"]:
        summary["status"] = "completed_with_errors"
    artifacts.write_csv(run_tag, "summarize_paper_results", rows)
    log_stage(
        logger,
        "%s complete run_id=%s status=%s summarized=%d skipped_existing=%d skipped_missing_pdf=%d skipped_not_accepted=%d failed=%d",
        SUMMARY_TAG,
        run_id,
        summary["status"],
        summary["summarized"],
        summary["skipped_existing"],
        summary["skipped_missing_pdf"],
        summary["skipped_not_accepted"],
        summary["failed"],
    )
    return summary


DEFAULT_OVERLAY_ROOT = "tasks/routed"


def triage_trading_papers(
    config_path: str | None = None,
    scopes_override: list[str] | None = None,
    overwrite: bool = False,
    pdf_text_max_chars: int = DEFAULT_PDF_TEXT_MAX_CHARS,
    overlay_root: str | None = None,
    routes_written_cap: int | None = None,
    log_practice: str = "triage_trading_papers",
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    llm_max_retries: int | None = None,
    temperature: float | None = None,
    timeout_sec: float | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Phase 1c: route trading-classified papers to curated tasks via LLM triage.

    Iterates every paper under ``cfg.papers.root_dir`` whose
    ``summary.task_family`` is in ``TRADING_FAMILIES``. For each, calls one LLM
    extraction to produce a ``TradingTriageRecord`` (persisted as
    ``trading_triage.json`` next to ``paper.json``) and renders a routed task
    overlay under ``overlay_root`` (default ``tasks/routed/<task_id>``).
    Writes ``paper.json["task"]`` with ``kind="curated_routed"`` so Phase 2-4
    skip the paper.

    ``routes_written_cap`` is an absolute scope-aware ceiling on the number
    of routed/partial_match papers (i.e. those that produce an overlay)
    per requested-scope union. Existing sidecars under each scope's
    ``paperN/trading_triage.json`` with ``status ∈ {routed, partial_match}``
    are subtracted from the cap; once the remaining budget is exhausted
    the loop continues but the LLM triage call is skipped for unseen
    trading papers (counted as ``skipped_written_cap``). No-match /
    novel_required outcomes produce no artifact and never consume the
    budget.

    Under ``overwrite=True`` existing sidecars are re-judged and re-written
    this run, so they are NOT subtracted — the budget is the full
    ``routes_written_cap`` (regenerate up to that many routes), not
    ``routes_written_cap`` minus existing.
    """
    if pdf_text_max_chars < 0:
        raise ValueError("pdf_text_max_chars must be >= 0")
    if routes_written_cap is not None and routes_written_cap < 0:
        raise ValueError("routes_written_cap must be >= 0")
    cfg = load_app_config(config_path)
    llm_cfg = resolve_phase_llm(
        cfg.llm,
        "triage",
        cli_provider=provider,
        cli_model=model,
        cli_max_tokens=max_tokens,
        cli_temperature=temperature,
        cli_llm_max_retries=llm_max_retries,
        cli_timeout_sec=timeout_sec,
    )
    all_scopes = load_scope_definitions(cfg.scopes.path)
    scopes = filter_scopes(all_scopes, scopes_override)
    scope_map = {scope.id: scope for scope in scopes}

    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
    run_tag = f"{run_id}_{log_practice}"
    run_dir = configure_practice_logging(
        cfg.logging.dir, log_practice, run_tag=run_tag, verbose=verbose
    )
    artifacts = ArtifactWriter(cfg.logging.dir)
    research_store = ResearchStore(cfg.papers.root_dir)
    downloader = PDFDownloader(cfg.download)
    from openfinai_pipeline.llm import LLMService

    llm = LLMService(llm_cfg)
    model_name = (
        f"{llm_cfg.provider}:{llm_cfg.providers[llm_cfg.provider].model}"
    )

    # Load catalog once; serialize once for prompt reuse across all papers.
    tasks_root = Path("tasks")
    catalog = load_curated_trading_catalog(tasks_root)
    if not catalog:
        raise RuntimeError(
            "no curated trading tasks found — expected at least one "
            f"triage_descriptor.toml with triage_eligible=true under "
            f"{tasks_root.resolve()}"
        )
    catalog_hash = compute_catalog_hash(catalog)
    catalog_version = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    catalog_serialized = serialize_catalog_for_prompt(catalog)
    overlay_dir_root = Path(overlay_root or DEFAULT_OVERLAY_ROOT)
    overlay_dir_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": run_id,
        "status": "completed",
        "catalog_version": catalog_version,
        "catalog_hash": catalog_hash,
        "catalog_size": len(catalog),
        "pdf_text_max_chars": pdf_text_max_chars,
        "routes_written_cap": routes_written_cap,
        "triaged": 0,
        "routed": 0,
        "partial_match": 0,
        "no_match": 0,
        "novel_required": 0,
        "skipped_not_trading": 0,
        "skipped_not_summarized": 0,
        "skipped_existing": 0,
        "skipped_missing_pdf": 0,
        "skipped_not_accepted": 0,
        "skipped_scope": 0,
        "skipped_written_cap": 0,
        "failed": 0,
    }
    rows: list[dict[str, Any]] = []
    task_dirs = [
        p
        for p in research_store.iter_task_dirs()
        if not scope_map or p.parent.name in scope_map
    ]

    # Existing routed sidecars consume the cross-scope budget unless overwrite
    # starts a fresh pass; count each sidecar once across the scope union.
    effective_cap: int | None = routes_written_cap
    existing_routes_in_scope = 0
    if routes_written_cap is not None:
        if scopes_override:
            scope_ids_for_count = [scope.id for scope in scopes]
        else:
            scope_ids_for_count = [
                scope_dir.name
                for scope_dir in research_store._root.iterdir()
                if scope_dir.is_dir()
            ]
        existing_routes_in_scope = sum(
            research_store.count_routes_for_scope(scope_id)
            for scope_id in scope_ids_for_count
        )
        effective_cap = (
            routes_written_cap
            if overwrite
            else max(0, routes_written_cap - existing_routes_in_scope)
        )
        log_stage(
            logger,
            "%s absolute cap: total_cap=%d existing_in_scope=%d overwrite=%s "
            "scope_filter=%s remaining_budget=%d",
            TRIAGE_TAG,
            routes_written_cap,
            existing_routes_in_scope,
            overwrite,
            scope_ids_for_count or "ALL",
            effective_cap,
        )
        if effective_cap <= 0:
            logger.info(
                "%s cap already met for new triage routes total_cap=%d "
                "existing_in_scope=%d; continuing in skip-write mode "
                "(existing sidecars still honored)",
                TRIAGE_TAG,
                routes_written_cap,
                existing_routes_in_scope,
            )

    log_stage(
        logger,
        "%s start run_id=%s candidate_papers=%d provider=%s model=%s catalog=%d cap=%s",
        TRIAGE_TAG,
        run_id,
        len(task_dirs),
        llm_cfg.provider,
        llm_cfg.providers[llm_cfg.provider].model,
        len(catalog),
        routes_written_cap if routes_written_cap is not None else "None",
    )
    logger.info(
        "%s start detail config=%s selected_scopes=%s papers_root=%s logs_dir=%s "
        "overwrite=%s pdf_text_max_chars=%s overlay_root=%s catalog_hash=%s "
        "routes_written_cap=%s",
        TRIAGE_TAG,
        config_path or "default",
        [scope.id for scope in scopes],
        cfg.papers.root_dir,
        run_dir,
        overwrite,
        pdf_text_max_chars,
        overlay_dir_root,
        catalog_hash,
        routes_written_cap if routes_written_cap is not None else "None",
    )

    routes_written_this_run = 0

    for paper_index, paper_dir in enumerate(task_dirs, start=1):
        rel_paper_dir = str(paper_dir.relative_to(research_store._root))
        scope_id = paper_dir.parent.name

        paper_json_path = paper_dir / "paper.json"
        if not paper_json_path.exists():
            summary["failed"] += 1
            rows.append({"paper_dir": rel_paper_dir, "status": "missing_paper_json"})
            continue

        paper_doc = json.loads(paper_json_path.read_text(encoding="utf-8"))

        decision_doc = paper_doc.get("decision") or {}
        if str(decision_doc.get("label", "")).strip().lower() != "accepted":
            summary["skipped_not_accepted"] += 1
            continue

        summary_payload = extract_summary_payload(paper_doc)
        if not summary_payload:
            summary["skipped_not_summarized"] += 1
            continue

        task_family = summary_payload.get("task_family")
        if not is_trading_family(task_family):
            summary["skipped_not_trading"] += 1
            continue

        triage_path = paper_dir / "trading_triage.json"
        if triage_path.exists() and not overwrite:
            summary["skipped_existing"] += 1
            rows.append(
                {
                    "paper_dir": rel_paper_dir,
                    "scope_id": scope_id,
                    "status": "skipped_existing",
                }
            )
            log_stage(logger, "%s   -> skip (existing triage)", TRIAGE_TAG)
            continue

        source_pdf = paper_dir / "source.pdf"
        if not source_pdf.exists():
            summary["skipped_missing_pdf"] += 1
            rows.append(
                {
                    "paper_dir": rel_paper_dir,
                    "scope_id": scope_id,
                    "status": "missing_source_pdf",
                }
            )
            log_stage(logger, "%s   -> skip (missing source.pdf)", TRIAGE_TAG)
            continue

        # Cap gate: skip without burning an LLM call once budget is exhausted.
        # Triage records persist only on success, so skipped papers remain
        # eligible on a later run with a raised cap.
        if effective_cap is not None and routes_written_this_run >= effective_cap:
            summary["skipped_written_cap"] += 1
            rows.append(
                {
                    "paper_dir": rel_paper_dir,
                    "scope_id": scope_id,
                    "task_family": task_family,
                    "status": "skipped_written_cap",
                }
            )
            log_stage(
                logger,
                "%s   -> skipped (written_cap_reached total_cap=%s effective_cap=%d)",
                TRIAGE_TAG,
                routes_written_cap,
                effective_cap,
            )
            continue

        paper_label = rel_paper_dir.replace("\\", "/")
        log_stage(
            logger,
            "%s [%d/%d] paper=%s task_family=%s",
            TRIAGE_TAG,
            paper_index,
            len(task_dirs),
            paper_label,
            task_family,
        )

        try:
            paper = PaperRecord.model_validate(paper_doc.get("paper") or {})
            task_name_raw = summary_payload.get("task_name")
            task_name = task_name_raw.strip() if isinstance(task_name_raw, str) else None
            record = triage_paper(
                paper_dir=paper_dir,
                paper_record=paper,
                catalog=catalog,
                catalog_version=catalog_version,
                catalog_hash=catalog_hash,
                llm=llm,
                downloader=downloader,
                model_name=model_name,
                pdf_text_max_chars=pdf_text_max_chars,
                catalog_serialized=catalog_serialized,
                task_name=task_name or None,
            )
            research_store.persist_trading_triage(
                paper_dir=paper_dir, record=record
            )
            summary["triaged"] += 1

            row = {
                "paper_dir": rel_paper_dir,
                "scope_id": scope_id,
                "task_family": task_family,
                "status": record.status.value,
                "confidence": record.confidence,
                "routed_from": record.routed_from or "",
                "recommended_task_id": record.recommended_task_id or "",
            }

            if record.status in (
                TradingTriageStatus.ROUTED,
                TradingTriageStatus.PARTIAL_MATCH,
            ):
                from openfinai_pipeline.papers._curated_catalog import find_descriptor

                descriptor = find_descriptor(catalog, record.routed_from or "")
                if descriptor is None:
                    raise TriageValidationIssue(
                        f"routed_from={record.routed_from!r} missing from catalog "
                        f"after triage validation (should be impossible)"
                    )
                overlay_path = render_overlay(
                    record=record,
                    descriptor=descriptor,
                    overlay_root=overlay_dir_root,
                    tasks_root=tasks_root,
                    paper_pdf=source_pdf,
                )
                triage_record_rel = str(triage_path.resolve())
                research_store.persist_paper_task(
                    paper_dir=paper_dir,
                    binding=PaperTaskBinding(
                        kind="curated_routed",
                        task_id=record.recommended_task_id,
                        task_dir=str(overlay_path.resolve()),
                        routed_from=record.routed_from,
                        triage_record_path=triage_record_rel,
                    ),
                )
                row["overlay_dir"] = str(overlay_path)
                routes_written_this_run += 1
                if record.status == TradingTriageStatus.ROUTED:
                    summary["routed"] += 1
                else:
                    summary["partial_match"] += 1
                log_stage(
                    logger,
                    "%s   -> %s overlay=%s confidence=%.2f",
                    TRIAGE_TAG,
                    record.status.value,
                    overlay_path,
                    record.confidence,
                )
            elif record.status == TradingTriageStatus.NO_MATCH:
                summary["no_match"] += 1
                log_stage(
                    logger,
                    "%s   -> no_match (gap=%s)",
                    TRIAGE_TAG,
                    truncate_oneline(record.gap_description or ""),
                )
            else:  # NOVEL_TASK_REQUIRED
                summary["novel_required"] += 1
                log_stage(
                    logger,
                    "%s   -> novel_task_required (features=%s)",
                    TRIAGE_TAG,
                    record.novel_features,
                )
            rows.append(row)
        except Exception as exc:
            summary["failed"] += 1
            rows.append(
                {
                    "paper_dir": rel_paper_dir,
                    "scope_id": scope_id,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            log_stage(
                logger,
                "%s   -> failed: %s",
                TRIAGE_TAG,
                truncate_oneline(str(exc)),
            )
            logger.exception(
                "%s failed paper_dir=%s error=%s", TRIAGE_TAG, rel_paper_dir, exc
            )

    if summary["failed"]:
        summary["status"] = "completed_with_errors"
    artifacts.write_csv(run_tag, "triage_trading_papers_results", rows)
    log_stage(
        logger,
        "%s complete run_id=%s status=%s triaged=%d routed=%d partial=%d "
        "no_match=%d novel=%d skipped_not_trading=%d skipped_existing=%d "
        "skipped_written_cap=%d failed=%d",
        TRIAGE_TAG,
        run_id,
        summary["status"],
        summary["triaged"],
        summary["routed"],
        summary["partial_match"],
        summary["no_match"],
        summary["novel_required"],
        summary["skipped_not_trading"],
        summary["skipped_existing"],
        summary["skipped_written_cap"],
        summary["failed"],
    )
    return summary


def import_manual_papers(
    config_path: str | None = None,
    mappings: list[str] | None = None,
    overwrite: bool = False,
    log_practice: str = "import_manual_papers",
) -> dict[str, Any]:
    cfg = load_app_config(config_path)
    all_scopes = load_scope_definitions(cfg.scopes.path)
    scope_map = {scope.id: scope for scope in all_scopes if scope.enabled}
    papers_root = Path(cfg.papers.root_dir)
    requested = _collect_manual_sources(
        mappings=mappings or [],
    )
    unknown_scope_ids = sorted({scope_id for scope_id, _ in requested if scope_id not in scope_map})
    if unknown_scope_ids:
        raise ValueError(
            f"Unknown or disabled scope ids in --mapping: {', '.join(unknown_scope_ids)}. "
            f"Add them to scopes.yaml and enable them before importing."
        )

    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
    run_tag = f"{run_id}_{log_practice}"
    run_dir = configure_practice_logging(cfg.logging.dir, log_practice, run_tag=run_tag, verbose=False)
    artifacts = ArtifactWriter(cfg.logging.dir)
    research_store = ResearchStore(cfg.papers.root_dir)
    downloader = PDFDownloader(cfg.download)
    staging_root = Path(cfg.benchmark.implementation.staging_dir).parent / "manual_import" / run_id

    summary = {
        "run_id": run_id,
        "status": "completed",
        "source_batches": len(requested),
        "imported": 0,
        "overwritten": 0,
        "skipped_existing": 0,
        "failed": 0,
    }
    rows: list[dict[str, Any]] = []

    log_stage(
        logger,
        "%s start run_id=%s mappings=%d overwrite=%s",
        IMPORT_TAG,
        run_id,
        len(requested),
        overwrite,
    )
    logger.info(
        "%s start detail config=%s papers_root=%s staging_root=%s mappings=%s",
        IMPORT_TAG,
        config_path or "default",
        cfg.papers.root_dir,
        staging_root,
        requested,
    )

    for scope_id, input_dir in requested:
        stage_scope_root = staging_root / scope_id
        stage_scope_root.mkdir(parents=True, exist_ok=True)
        staged_batch_dir, staged_entries = _stage_manual_source_batch(
            input_dir=input_dir,
            scope_id=scope_id,
            stage_scope_root=stage_scope_root,
            papers_root=papers_root,
        )
        logger.info(
            "%s scope=%s input_dir=%s staged_batch_dir=%s candidates=%d",
            IMPORT_TAG,
            scope_id,
            input_dir,
            staged_batch_dir,
            len(staged_entries),
        )
        for source_path, staged_path in staged_entries:
            row = {
                "scope_id": scope_id,
                "source_path": str(source_path),
                "staged_path": str(staged_path),
                "paper_dir": "",
                "status": "",
                "error": "",
            }
            try:
                file_hash = _sha256_file(staged_path)
                title = _extract_manual_title(staged_path)
                paper_id = _manual_paper_id(
                    title=title,
                    scope_id=scope_id,
                    research_store=research_store,
                )
                existing_dir = research_store.find_task_dir_by_raw_payload_hash(
                    scope_id=scope_id,
                    raw_payload_hash=file_hash,
                )
                if existing_dir is not None and not overwrite:
                    summary["skipped_existing"] += 1
                    row["paper_dir"] = str(existing_dir.relative_to(Path(cfg.papers.root_dir)))
                    row["status"] = "skipped_existing"
                    rows.append(row)
                    continue

                abstract = _extract_manual_abstract(staged_path, downloader)
                paper = PaperRecord(
                    paper_id=paper_id,
                    source="manual",
                    title=title,
                    abstract=abstract,
                    raw_payload_hash=file_hash,
                    scope_ids=[scope_id],
                    status=PaperStatus.ACCEPTED,
                )
                decision = JudgeDecision(
                    paper_id=paper_id,
                    scope_id=scope_id,
                    label=JudgeLabel.ACCEPTED,
                    score_0_10=10.0,
                    reasons="Manually imported local PDF.",
                    confidence_0_1=1.0,
                    model="manual_import",
                    raw_response={"source_path": str(source_path), "staged_path": str(staged_path)},
                )
                paper_dir = research_store.persist_accepted_paper(
                    run_id=run_id,
                    scope_id=scope_id,
                    paper=paper,
                    decision=decision,
                    source_pdf_path=str(staged_path),
                    preserve_source_pdf=True,
                )
                row["paper_dir"] = paper_dir
                if existing_dir is not None:
                    summary["overwritten"] += 1
                    row["status"] = "overwritten"
                else:
                    summary["imported"] += 1
                    row["status"] = "imported"
                rows.append(row)
            except Exception as exc:
                summary["failed"] += 1
                row["status"] = "failed"
                row["error"] = str(exc)
                rows.append(row)
                logger.exception("%s failed scope=%s source=%s error=%s", IMPORT_TAG, scope_id, source_path, exc)

    if summary["failed"]:
        summary["status"] = "completed_with_errors"
    artifacts.write_csv(run_tag, "manual_import_results", rows)
    log_stage(
        logger,
        "%s complete run_id=%s status=%s imported=%d overwritten=%d skipped_existing=%d failed=%d run_dir=%s",
        IMPORT_TAG,
        run_id,
        summary["status"],
        summary["imported"],
        summary["overwritten"],
        summary["skipped_existing"],
        summary["failed"],
        run_dir,
    )
    return summary


def _effective_sift_budget(judge_cfg: JudgeConfig, per_scope_limit: int) -> int:
    """Resolve the per-scope Stage-2 (sift) LLM call budget.

    The fraction scales the sift budget with the scrape size so raising
    ``--limit`` automatically grants more sift calls. ``max_llm_calls_per_scope``
    remains a hard ceiling to bound cost on very large scrapes; setting the
    fraction to ``0`` disables scaling and uses the explicit ceiling alone.
    """
    fraction = judge_cfg.llm_judge_fraction
    if fraction and fraction > 0:
        scaled = max(1, int(per_scope_limit * fraction))
        return min(judge_cfg.max_llm_calls_per_scope, scaled)
    return judge_cfg.max_llm_calls_per_scope


def _collect_manual_sources(
    *,
    mappings: list[str],
) -> list[tuple[str, Path]]:
    requested = _parse_manual_mappings(mappings)
    deduped: list[tuple[str, Path]] = []
    seen: set[tuple[str, Path]] = set()
    for scope_id, input_dir in requested:
        key = (scope_id, input_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append((scope_id, input_dir.resolve()))
    if not deduped:
        raise ValueError("Provide at least one --mapping <scope_id>=<folder>.")
    return deduped


def _parse_manual_mappings(mappings: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for raw in mappings:
        scope_id, sep, folder = raw.partition("=")
        scope_id = scope_id.strip()
        folder = folder.strip()
        if not sep or not scope_id or not folder:
            raise ValueError(f"Invalid mapping '{raw}'. Expected <scope_id>=<folder>.")
        input_dir = Path(folder).expanduser()
        if not input_dir.is_absolute():
            input_dir = input_dir.resolve()
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Mapping folder does not exist or is not a directory: {folder}")
        parsed.append((scope_id, input_dir))
    return parsed


def _list_manual_pdf_candidates(input_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for child in sorted(input_dir.rglob("*.pdf")):
        if not child.is_file():
            continue
        if _path_has_paper_dir_ancestor(child, stop_at=input_dir):
            continue
        candidates.append(child.resolve())
    return candidates


def _build_stage_filename(index: int, original_name: str) -> str:
    safe_name = re.sub(r"[^\w.\-]+", "_", original_name).strip("._") or "paper.pdf"
    return f"{index:04d}__{safe_name}"


def _stage_manual_source_batch(
    *,
    input_dir: Path,
    scope_id: str,
    stage_scope_root: Path,
    papers_root: Path,
) -> tuple[Path, list[tuple[Path, Path]]]:
    input_dir = input_dir.resolve()
    if _should_stage_whole_directory(input_dir, papers_root=papers_root):
        return _stage_whole_manual_directory(
            input_dir=input_dir,
            scope_id=scope_id,
            stage_scope_root=stage_scope_root,
        )
    return _stage_manual_pdf_files(
        input_dir=input_dir,
        stage_scope_root=stage_scope_root,
    )


def _stage_whole_manual_directory(
    *,
    input_dir: Path,
    scope_id: str,
    stage_scope_root: Path,
) -> tuple[Path, list[tuple[Path, Path]]]:
    pdf_paths = _list_manual_pdf_candidates(input_dir)
    staged_dir = _unique_stage_dir(stage_scope_root / _stage_folder_name(input_dir, scope_id))
    staged_dir.parent.mkdir(parents=True, exist_ok=True)
    pdf_rel_paths = [path.relative_to(input_dir) for path in pdf_paths]
    shutil.move(str(input_dir), str(staged_dir))
    staged_entries = [
        (input_dir / rel_path, staged_dir / rel_path)
        for rel_path in pdf_rel_paths
    ]
    return staged_dir, staged_entries


def _stage_manual_pdf_files(
    *,
    input_dir: Path,
    stage_scope_root: Path,
) -> tuple[Path, list[tuple[Path, Path]]]:
    pdf_paths = _list_manual_pdf_candidates(input_dir)
    staged_dir = _unique_stage_dir(stage_scope_root / _stage_folder_name(input_dir))
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_entries: list[tuple[Path, Path]] = []
    for index, source_path in enumerate(pdf_paths, start=1):
        staged_path = staged_dir / _build_stage_filename(index, source_path.name)
        shutil.move(str(source_path), str(staged_path))
        staged_entries.append((source_path, staged_path))
    return staged_dir, staged_entries


def _path_has_paper_dir_ancestor(path: Path, *, stop_at: Path) -> bool:
    current = path.parent
    stop_at = stop_at.resolve()
    while True:
        if re.fullmatch(r"paper\d+", current.name):
            return True
        if current == stop_at or current.parent == current:
            return False
        current = current.parent


def _should_stage_whole_directory(input_dir: Path, *, papers_root: Path) -> bool:
    try:
        input_dir.relative_to(papers_root)
        return False
    except ValueError:
        return True


def _stage_folder_name(input_dir: Path, scope_id: str | None = None) -> str:
    parts = [part for part in input_dir.parts[-3:] if part not in {"/", ""}]
    label = "__".join(parts) or input_dir.name or "manual_import"
    label = re.sub(r"[^\w.\-]+", "_", label).strip("._") or "manual_import"
    if scope_id:
        return f"{label}__{scope_id}"
    return label


def _unique_stage_dir(path: Path) -> Path:
    if not path.exists():
        return path
    parent = path.parent
    stem = path.name
    for index in range(2, 10_000):
        candidate = parent / f"{stem}_{index}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to allocate unique staging directory for {path}")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _manual_paper_id(
    *,
    title: str,
    scope_id: str,
    research_store: ResearchStore,
) -> str:
    base_title = re.sub(r"\s+", " ", str(title).strip())
    if not base_title:
        base_title = f"Untitled Manual Import ({scope_id})"
    existing_ids = set(research_store.list_paper_ids(scope_id=scope_id))
    if base_title not in existing_ids:
        return base_title
    for index in range(2, 10_000):
        candidate = f"{base_title} ({index})"
        if candidate not in existing_ids:
            return candidate
    raise RuntimeError(f"Unable to allocate unique manual paper_id for title: {base_title}")


def _extract_manual_title(pdf_path: Path) -> str:
    content_title = _extract_manual_title_from_pdf_content(pdf_path)
    if content_title:
        return content_title
    try:
        import fitz  # type: ignore

        doc = fitz.open(pdf_path)
        try:
            metadata = doc.metadata or {}
            title = str(metadata.get("title") or "").strip()
        finally:
            doc.close()
        if title and title.lower() != "untitled":
            return title
    except Exception:
        pass
    return ""


def _extract_manual_title_from_pdf_content(pdf_path: Path) -> str:
    page_text = _extract_first_page_text(pdf_path)
    if not page_text:
        return ""
    return _extract_title_from_text(page_text)


def _extract_first_page_text(pdf_path: Path) -> str:
    try:
        proc = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "1", "-nopgbrk", str(pdf_path), "-"],
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        proc = None
    if proc is not None and proc.returncode == 0:
        text = (proc.stdout or "").strip()
        if text:
            return text

    try:
        import fitz  # type: ignore

        doc = fitz.open(pdf_path)
        try:
            if doc.page_count <= 0:
                return ""
            text = doc.load_page(0).get_text("text") or ""
        finally:
            doc.close()
        return text.strip()
    except Exception:
        return ""


def _extract_title_from_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in normalized.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""

    stop_index = len(lines)
    for idx, line in enumerate(lines[:40]):
        lower = line.lower()
        if re.match(r"^(abstract|summary)\b", lower):
            stop_index = idx
            break
        if re.match(r"^(keywords?|index terms)\b", lower):
            stop_index = idx
            break
        if re.match(
            r"^(?:\d+(?:\.\d+)*)?\s*(introduction|background|related work|methods?|data|experiments?|results?|conclusion)\b",
            lower,
        ):
            stop_index = idx
            break

    candidate_lines = lines[:stop_index]
    filtered: list[str] = []
    for line in candidate_lines[:12]:
        if _looks_like_non_title_line(line):
            continue
        filtered.append(line)
        if len(filtered) >= 3:
            break

    if not filtered:
        for line in lines[:8]:
            if _looks_like_non_title_line(line):
                continue
            filtered.append(line)
            if len(filtered) >= 3:
                break

    if not filtered:
        return ""

    title = " ".join(filtered)
    title = re.sub(r"\s+", " ", title).strip(" -:|,;")
    if len(title) < 6:
        return ""
    return title[:300]


def _looks_like_non_title_line(line: str) -> bool:
    lower = line.lower().strip()
    if not lower:
        return True
    if len(lower) > 220:
        return True
    if re.fullmatch(r"\d+", lower):
        return True
    if "@" in lower:
        return True
    if lower.startswith("http://") or lower.startswith("https://") or "www." in lower:
        return True
    if re.search(r"\barxiv\b|\bdoi\b|issn|isbn|working paper|preprint|accepted at|proceedings|conference|journal\b", lower):
        return True
    if re.match(r"^(abstract|summary|keywords?|index terms)\b", lower):
        return True
    if re.match(r"^(?:\d+(?:\.\d+)*)?\s*(introduction|background|related work|methods?|data|experiments?|results?|conclusion)\b", lower):
        return True
    if re.search(r"\b(university|department|school|college|laboratory|lab|institute|faculty)\b", lower):
        return True
    if re.search(r"\b(corresponding author|acknowledg)\b", lower):
        return True
    if re.fullmatch(r"[A-Za-z.\- ]+(?:, [A-Za-z.\- ]+){1,6}", line) and len(line.split()) <= 18:
        return True
    return False


def _extract_manual_abstract(pdf_path: Path, downloader: PDFDownloader) -> str:
    excerpt = downloader.extract_excerpt_from_path(pdf_path, max_chars=20000) or ""
    if not excerpt:
        return ""
    return _extract_abstract_from_text(excerpt)


def _extract_abstract_from_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    start_match = re.search(r"(?im)^\s*abstract\b[:\s-]*", normalized)
    if not start_match:
        return ""
    remainder = normalized[start_match.end():].lstrip()
    stop_match = re.search(
        r"(?im)^\s*(?:keywords?\b|index terms\b|"
        r"(?:\d+(?:\.\d+)*)?\s*(?:introduction|background|related work|methods?|data|experiments?|results?|conclusion|references)\b)",
        remainder,
    )
    abstract = remainder[:stop_match.start()] if stop_match else remainder
    abstract = re.sub(r"\s+", " ", abstract).strip(" -:\n\t")
    return abstract[:4000].strip()


def _new_summary(scopes: list[ScopeDefinition]) -> RunSummary:
    return RunSummary(
        run_id=datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S"),
        started_at=datetime.now(tz=timezone.utc).isoformat(),
        scope_ids=[s.id for s in scopes],
    )


def _collect_scope_papers(
    *,
    summary: RunSummary,
    arxiv: ArxivClient,
    crossref: CrossrefClient,
    s2: SemanticScholarClient,
    scope: ScopeDefinition,
    since_date: datetime | None,
    until_date: datetime,
    per_scope_limit: int,
) -> list[PaperRecord]:
    logger.info(
        "%s scope=%s collect from=arxiv queries=%d categories=%d",
        PHASE_TAG,
        scope.id,
        len(scope.queries),
        len(scope.categories),
    )
    papers = arxiv.scrape_scope(
        scope,
        since_date=since_date,
        until_date=until_date,
        max_papers=per_scope_limit,
    )
    summary.total_scraped += len(papers)

    logger.info(
        "%s scope=%s collect_complete arxiv_papers=%d",
        PHASE_TAG,
        scope.id,
        len(papers),
    )

    papers = crossref.enrich_doi(papers)
    papers = s2.enrich(papers)

    citation_enriched = sum(1 for p in papers if p.citation_count is not None)
    s2_enriched = sum(
        1
        for p in papers
        if p.semantic_scholar_id
        or p.influential_citation_count is not None
        or bool(p.venue)
        or bool(p.journal_name)
        or bool(p.publication_types)
    )
    summary.total_enriched += citation_enriched
    logger.info(
        "%s scope=%s enrich_complete citation_enriched=%d semantic_scholar_enriched=%d",
        PHASE_TAG,
        scope.id,
        citation_enriched,
        s2_enriched,
    )

    return papers


def _apply_seen_dedupe(
    *,
    summary: RunSummary,
    research_store: ResearchStore,
    scope: ScopeDefinition,
    papers: list[PaperRecord],
    overwrite: bool,
) -> list[PaperRecord]:
    """Drop papers already judged in a prior run for this scope.

    Reads the per-scope ``index.json`` once: papers whose ``paper_id`` is
    in ``entries`` (accepted) or ``seen_rejected`` are filtered out so the
    expensive prefilter/sift/download stages don't run on them. Bypassed
    entirely when ``overwrite`` is True.

    Counters land on the run summary so the rerun-skip count is visible
    in the printed line and the ``scrape_summary.json`` artifact.
    """
    if overwrite:
        log_stage(
            logger,
            "%s scope=%s dedupe overwrite=True (re-judging all %d collected papers)",
            PHASE_TAG,
            scope.id,
            len(papers),
        )
        return papers

    accepted_seen, rejected_seen = research_store.list_seen_paper_ids(
        scope_id=scope.id
    )
    if not accepted_seen and not rejected_seen:
        logger.info(
            "%s scope=%s dedupe no-prior-judgments collected=%d",
            PHASE_TAG,
            scope.id,
            len(papers),
        )
        return papers

    skipped_accepted = 0
    skipped_rejected = 0
    filtered: list[PaperRecord] = []
    for paper in papers:
        if paper.paper_id in accepted_seen:
            skipped_accepted += 1
            continue
        if paper.paper_id in rejected_seen:
            skipped_rejected += 1
            continue
        filtered.append(paper)

    summary.total_skipped_seen_accepted += skipped_accepted
    summary.total_skipped_seen_rejected += skipped_rejected
    log_stage(
        logger,
        "%s scope=%s dedupe collected=%d skipped_seen_accepted=%d "
        "skipped_seen_rejected=%d remaining=%d "
        "(pass --overwrite to re-judge)",
        PHASE_TAG,
        scope.id,
        len(papers),
        skipped_accepted,
        skipped_rejected,
        len(filtered),
    )
    return filtered


def _prefetch_excerpts(
    *, cfg: AppConfig, downloader: PDFDownloader, papers: list[PaperRecord]
) -> dict[str, str]:
    if not cfg.download.enabled:
        logger.info("%s excerpt_prefetch disabled=true", PHASE_TAG)
        return {}
    excerpts: dict[str, str] = {}
    eligible = 0
    total = len(papers)
    for index, paper in enumerate(papers, start=1):
        if not paper.pdf_url:
            logger.info(
                "%s excerpt %d/%d -> skip (no pdf_url)",
                PHASE_TAG, index, total,
            )
            continue
        eligible += 1
        excerpt = downloader.extract_excerpt(paper.pdf_url)
        if excerpt:
            excerpts[paper.paper_id] = excerpt
            logger.info(
                "%s excerpt %d/%d -> ok (%s)",
                PHASE_TAG, index, total, fmt_chars(len(excerpt)),
            )
        else:
            logger.info(
                "%s excerpt %d/%d -> fail",
                PHASE_TAG, index, total,
            )
    logger.info(
        "%s excerpt_prefetch eligible=%d extracted=%d",
        PHASE_TAG,
        eligible,
        len(excerpts),
    )
    return excerpts


def _judge_scope(
    *,
    summary: RunSummary,
    judge: JudgeService,
    scope: ScopeDefinition,
    papers: list[PaperRecord],
    excerpts: dict[str, str],
    sift_budget: int,
) -> list[JudgeDecision]:
    decisions = judge.judge_scope(
        scope, papers, pdf_excerpts=excerpts, sift_budget=sift_budget
    )
    summary.total_judged += sum(
        1
        for d in decisions
        if not d.model.startswith("prefilter") and d.model != "budget_cap"
    )
    logger.info(
        "%s scope=%s judge_complete total=%d accepted=%d rejected=%d sift_calls=%d",
        PHASE_TAG,
        scope.id,
        len(decisions),
        sum(1 for decision in decisions if decision.label == JudgeLabel.ACCEPTED),
        sum(1 for decision in decisions if decision.label != JudgeLabel.ACCEPTED),
        sum(
            1
            for decision in decisions
            if not decision.model.startswith("prefilter")
            and decision.model != "budget_cap"
        ),
    )
    return decisions


def _download_accepted_pdfs(
    *,
    cfg: AppConfig,
    summary: RunSummary,
    judge: JudgeService,
    downloader: PDFDownloader,
    scope: ScopeDefinition,
    papers: list[PaperRecord],
    decisions: list[JudgeDecision],
    prefetch_excerpts: dict[str, str],
    per_scope_max_accepts: int | None,
    research_store: "ResearchStore",
    overwrite: bool = False,
) -> tuple[list[JudgeDecision], dict[str, str]]:
    if not cfg.download.enabled:
        logger.info("%s scope=%s download disabled=true", PHASE_TAG, scope.id)
        return decisions, {}

    # Compute effective accept cap: absolute total minus existing accepted
    # papers. Under --overwrite the dedupe filter is bypassed upstream, so
    # existing accepts are re-collected and re-judged this run; subtracting
    # them here too would double-count and force the cap to downgrade
    # previously-accepted papers. Overwrite is a fresh slate, so the budget
    # is the full cap.
    effective_max_accepts = per_scope_max_accepts
    if per_scope_max_accepts is not None:
        existing_accepted = research_store.count_accepted_for_scope(scope.id)
        effective_max_accepts = (
            per_scope_max_accepts
            if overwrite
            else max(0, per_scope_max_accepts - existing_accepted)
        )
        logger.info(
            "%s scope=%s absolute accept cap: total_cap=%d existing=%d overwrite=%s remaining_budget=%d",
            PHASE_TAG,
            scope.id,
            per_scope_max_accepts,
            existing_accepted,
            overwrite,
            effective_max_accepts,
        )
        if effective_max_accepts <= 0:
            logger.info(
                "%s scope=%s accept cap already met, rejecting all new papers total_cap=%d existing=%d",
                PHASE_TAG,
                scope.id,
                per_scope_max_accepts,
                existing_accepted,
            )
            for d in decisions:
                if d.label == JudgeLabel.ACCEPTED:
                    d.label = JudgeLabel.REJECTED
                    d.reasons = (
                        (d.reasons or "")
                        + f" Accept cap already met (existing={existing_accepted}, cap={per_scope_max_accepts})."
                    )
            return decisions, {}

    if effective_max_accepts is not None and effective_max_accepts > 0:
        decisions = judge.enforce_downloadable_accept_cap(
            scope=scope,
            papers=papers,
            decisions=decisions,
            pdf_excerpts=prefetch_excerpts,
            max_accepts=effective_max_accepts,
        )

    paper_map = {p.paper_id: p for p in papers}
    downloaded: dict[str, str] = {}
    success = 0
    accepted_decisions = [
        decision for decision in decisions if decision.label == JudgeLabel.ACCEPTED
    ]
    accepted = len(accepted_decisions)

    for index, decision in enumerate(accepted_decisions, start=1):
        paper = paper_map.get(decision.paper_id)
        if not paper or not paper.pdf_url:
            logger.info(
                "%s download %d/%d -> skip (no pdf_url)",
                PHASE_TAG, index, accepted,
            )
            continue
        result = downloader.download(decision.paper_id, paper.pdf_url)
        if not result:
            logger.info(
                "%s download %d/%d -> fail",
                PHASE_TAG, index, accepted,
            )
            continue
        file_path, written, _, _ = result
        downloaded[decision.paper_id] = file_path
        summary.total_downloaded += 1
        success += 1
        logger.info(
            "%s download %d/%d -> ok (%s)",
            PHASE_TAG, index, accepted, fmt_bytes(written),
        )

    logger.info(
        "%s scope=%s download_complete accepted_candidates=%d pdf_downloaded=%d",
        PHASE_TAG,
        scope.id,
        accepted,
        success,
    )
    return decisions, downloaded


def _persist_scope_results(
    *,
    summary: RunSummary,
    research_store: ResearchStore,
    scope: ScopeDefinition,
    run_id: str,
    papers: list[PaperRecord],
    decisions: list[JudgeDecision],
    downloaded_pdf_paths: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paper_map = {p.paper_id: p for p in papers}
    judgment_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []

    for decision in decisions:
        paper = paper_map.get(decision.paper_id)
        _apply_decision_to_paper(paper, decision.label)
        _count_decision(summary, decision)
        if paper is None:
            continue
        judgment_rows.append(
            _build_judgment_row(
                run_id=run_id,
                scope_id=scope.id,
                paper=paper,
                decision=decision,
                pdf_download_success=decision.paper_id in downloaded_pdf_paths,
            )
        )

    accepted_count = 0
    persisted_dirs: list[str] = []
    for decision in decisions:
        if decision.label != JudgeLabel.ACCEPTED:
            continue
        paper = paper_map.get(decision.paper_id)
        if paper is None:
            continue
        paper_dir = research_store.persist_accepted_paper(
            run_id=run_id,
            scope_id=scope.id,
            paper=paper,
            decision=decision,
            source_pdf_path=downloaded_pdf_paths.get(decision.paper_id),
        )
        accepted_rows.append(_build_accepted_row(paper, scope_id=scope.id))
        accepted_count += 1
        persisted_dirs.append(paper_dir)

    # Log rejected decisions so subsequent scrape runs (no --overwrite) skip
    # them upfront. record_rejected enforces entries/seen_rejected disjoint.
    rejected_items: list[tuple[str, str, str]] = []
    for decision in decisions:
        if decision.label == JudgeLabel.ACCEPTED:
            continue
        if decision.paper_id not in paper_map:
            continue
        rejected_items.append(
            (decision.paper_id, decision.model, decision.reasons or "")
        )
    if rejected_items:
        research_store.record_rejected(
            scope_id=scope.id,
            run_id=run_id,
            items=rejected_items,
        )

    summary.total_tasks_constructed += accepted_count
    logger.info(
        "%s scope=%s persist_complete accepted=%d rejected_logged=%d papers_root=%s",
        PHASE_TAG,
        scope.id,
        accepted_count,
        len(rejected_items),
        research_store._root,
    )
    return judgment_rows, accepted_rows


def _apply_decision_to_paper(paper: PaperRecord | None, label: JudgeLabel) -> None:
    if paper is None:
        return
    paper.status = (
        PaperStatus.ACCEPTED if label == JudgeLabel.ACCEPTED else PaperStatus.REJECTED
    )


def _count_decision(summary: RunSummary, decision: JudgeDecision) -> None:
    if decision.model.startswith("prefilter"):
        summary.total_prefiltered += 1
    if decision.label == JudgeLabel.ACCEPTED:
        summary.total_accepted += 1
    else:
        summary.total_rejected += 1


def _build_judgment_row(
    *,
    run_id: str,
    scope_id: str,
    paper: PaperRecord,
    decision: JudgeDecision,
    pdf_download_success: bool,
) -> dict[str, Any]:
    evidence = (
        decision.raw_response.get("evidence")
        if isinstance(decision.raw_response, dict)
        else {}
    )
    if not isinstance(evidence, dict):
        evidence = {}
    return {
        "run_id": run_id,
        "scope_id": scope_id,
        "paper_id": paper.paper_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "doi": paper.doi,
        "citation_count": paper.citation_count,
        "influential_citation_count": paper.influential_citation_count,
        "venue": paper.venue,
        "journal_name": paper.journal_name,
        "publication_types_json": json.dumps(paper.publication_types),
        "peer_reviewed": int(paper.peer_reviewed),
        "label": decision.label.value,
        "score_0_10": decision.score_0_10,
        "confidence_0_1": decision.confidence_0_1,
        "model": decision.model,
        "prefilter_passed": int(paper.prefilter_passed),
        "prefilter_score": paper.prefilter_score,
        "prefilter_result": "pass" if paper.prefilter_passed else "reject",
        "pdf_download_success": pdf_download_success,
        "reasons_json": json.dumps(str(decision.reasons or "")),
        "evidence_experiments_json": json.dumps(str(evidence.get("experiments") or "")),
        "evidence_datasets_json": json.dumps(str(evidence.get("datasets") or "")),
        "evidence_metrics_json": json.dumps(str(evidence.get("metrics") or "")),
    }


def _build_accepted_row(paper: PaperRecord, scope_id: str) -> dict[str, Any]:
    return {
        "scope_id": scope_id,
        "Title": paper.title or "",
        "Publication Date": paper.published_at or "",
        "Author(s)": ", ".join(str(a) for a in paper.authors if a),
        "Abstract": paper.abstract or "",
        "Link": paper.pdf_url or "",
        "DOI": paper.doi or "",
        "Citation Count": paper.citation_count
        if paper.citation_count is not None
        else "",
        "Influential Citation Count": (
            paper.influential_citation_count
            if paper.influential_citation_count is not None
            else ""
        ),
        "Venue": paper.venue or "",
        "Journal": paper.journal_name or "",
        "Publication Types": ", ".join(paper.publication_types),
        "Peer Reviewed": "TRUE" if paper.peer_reviewed else "FALSE",
        "Relevant": "TRUE",
    }


def _sort_judgment_rows(rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda item: float(item.get("score_0_10") or 0.0), reverse=True)
    rows.sort(key=lambda item: str(item.get("scope_id") or ""))


def _sort_accepted_rows(rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda item: str(item.get("Title") or ""))
    rows.sort(key=lambda item: str(item.get("Publication Date") or ""), reverse=True)
