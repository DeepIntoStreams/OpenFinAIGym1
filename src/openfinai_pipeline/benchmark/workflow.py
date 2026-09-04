"""
Phase 4 workflow: construct canonical benchmark task packages.

Entry point: construct_tasks()
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from openfinai_pipeline.benchmark.builders.task_builder import (
    AssemblyGroundTruthError,
    _build_artifact_plan,
    _infer_interaction_model,
    _read_paper_context,
    _staging_folder_name,
    _validate_artifact_plan_for_assembly,
    assemble_evaluator,
    can_reuse_existing_task,
    enrich_candidate_from_phases,
    implement_task,
    load_task_candidates,
)
from openfinai_pipeline.benchmark.loader import LoaderResult, generate_task_loader
from openfinai_pipeline.benchmark.smoke_validation import (
    dry_fire_evaluator,
    short_smoke_reason,
    validate_installed_task_bundle,
    write_dry_fire_failure_sidecar,
    write_validation_failure_sidecar,
)
from openfinai_pipeline.corpus.dataset_filter import check_paper_datasets_ready
from openfinai_pipeline.llm import LLMService
from openfinai_pipeline.papers.pdf_downloader import PDFDownloader
from openfinai_pipeline.papers.schemas import PaperTaskBinding
from openfinai_pipeline.papers.triage import TRADING_FAMILIES
from openfinai_pipeline.settings import load_app_config, resolve_phase_llm
from openfinai_pipeline.utils.logging import (
    configure_practice_logging,
    log_detail,
    log_stage,
    truncate_oneline,
)
from openfinai_pipeline.utils.sandbox import Sandbox

logger = logging.getLogger(__name__)
PHASE_TAG = "[phase4:benchmark]"


def _rel_path(path: Path | str) -> str:
    """Return *path* relative to the cwd in POSIX form, or absolute POSIX if outside cwd."""
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _short_dry_fire_reason(errors: list[str]) -> str:
    """Compact single-line reason for the dry-fire failure stage marker.

    Picks the first error string (typically ``reward_dispatch_missing:<X>``)
    and appends ``(+N more)`` if there are additional errors.
    """
    if not errors:
        return "unknown"
    first = truncate_oneline(str(errors[0]))
    if len(errors) > 1:
        return f"{first} (+{len(errors) - 1} more)"
    return first


def _update_paper_task_record(
    paper_dir: Path | str,
    *,
    installed_task_dir: Path,
    generated: bool,
) -> None:
    """Stamp paper.json["task"] with a PaperTaskBinding(kind="auto_generated").

    The unified ``kind`` discriminator lets downstream consumers dispatch on
    auto_generated vs curated_routed (Phase 1c) without parallel field checks.
    """
    paper_json_path = Path(paper_dir) / "paper.json"
    if not paper_json_path.exists():
        return
    try:
        paper_doc = json.loads(paper_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    binding = PaperTaskBinding(
        kind="auto_generated",
        task_dir=str(installed_task_dir.resolve()),
        generated=bool(generated),
    )
    paper_doc["task"] = binding.model_dump(mode="json", exclude_none=True)
    paper_json_path.write_text(
        json.dumps(paper_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def construct_tasks(
    config_path: str | None = None,
    *,
    tasks_written_cap: int | None = None,
    scope_ids: list[str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    llm_max_retries: int | None = None,
    tool_max_turns: int | None = None,
    task_generation_rounds: int | None = None,
    loader_generation_rounds: int | None = None,
    timeout_sec: float | None = None,
    overwrite: bool = False,
    retry_failed_attempts: bool = False,
    verbose: bool = False,
) -> dict[str, int]:
    """
    Construct canonical benchmark task packages from harvested papers.

    For each task candidate:
      1. Dedup against existing installed tasks (skip if duplicate)
      2. LLM code generation -> static analysis -> pytest -> error-driven retry
      3. Install passing tasks to the configured benchmark install dir
    """
    cfg = load_app_config(config_path)
    cli_kwargs = dict(
        cli_provider=provider,
        cli_model=model,
        cli_temperature=temperature,
        cli_max_tokens=max_tokens,
        cli_llm_max_retries=llm_max_retries,
        cli_timeout_sec=timeout_sec,
        cli_tool_max_turns=tool_max_turns,
    )
    task_llm_cfg = resolve_phase_llm(cfg.llm, "task", **cli_kwargs)
    loader_llm_cfg = resolve_phase_llm(cfg.llm, "loader", **cli_kwargs)
    resolved_task_generation_rounds = int(
        task_generation_rounds
        if task_generation_rounds is not None
        else cfg.benchmark.task_generation_rounds
    )
    if resolved_task_generation_rounds < 1:
        raise ValueError("task_generation_rounds must be >= 1")
    resolved_loader_generation_rounds = int(
        loader_generation_rounds
        if loader_generation_rounds is not None
        else cfg.benchmark.loader_generation_rounds
    )
    if resolved_loader_generation_rounds < 1:
        raise ValueError("loader_generation_rounds must be >= 1")

    run_tag = (
        f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}_construct_tasks"
    )
    run_dir = configure_practice_logging(
        cfg.logging.dir, "construct_tasks", run_tag=run_tag, verbose=verbose
    )

    llm = LLMService(task_llm_cfg)
    # Loader codegen uses its own per-phase override (cfg.llm.per_phase.loader);
    # share the LLMService with task assembly when both phases resolve to the
    # same effective config to avoid spinning up redundant clients.
    if loader_llm_cfg.model_dump() == task_llm_cfg.model_dump():
        loader_llm = llm
    else:
        loader_llm = LLMService(loader_llm_cfg)
    impl_cfg = cfg.benchmark.implementation

    research_root = Path(cfg.papers.root_dir)
    staging_root = Path(impl_cfg.staging_dir)
    tasks_root = Path(impl_cfg.tasks_install_dir)

    staging_root.mkdir(parents=True, exist_ok=True)
    tasks_root.mkdir(parents=True, exist_ok=True)

    sandbox = Sandbox(timeout_seconds=impl_cfg.sandbox_timeout_sec)
    pdf_downloader = PDFDownloader(cfg.download)

    # Resolve Phase 2/3 artifact directories
    datasets_root = Path(cfg.datasets.root_dir)
    curated_datasets_root = Path(cfg.datasets.curated_root_dir)
    rewards_root = Path(cfg.rewards.root_dir)
    generated_dir = Path(cfg.rewards.generated_dir)
    log_stage(
        logger,
        "%s start scope_filter=%s tasks_written_cap=%s task_model=%s loader_model=%s",
        PHASE_TAG,
        scope_ids or "ALL",
        tasks_written_cap if tasks_written_cap is not None else "ALL",
        task_llm_cfg.providers[task_llm_cfg.provider].model,
        loader_llm_cfg.providers[loader_llm_cfg.provider].model,
    )
    logger.info(
        "%s   start detail config=%s papers=%s staging=%s install=%s logs=%s task_rounds=%d loader_rounds=%d overwrite=%s retry_failed=%s",
        PHASE_TAG,
        config_path or "default",
        _rel_path(research_root),
        _rel_path(staging_root),
        _rel_path(tasks_root),
        _rel_path(run_dir),
        resolved_task_generation_rounds,
        resolved_loader_generation_rounds,
        overwrite,
        retry_failed_attempts,
    )

    # Load candidates
    candidates = load_task_candidates(research_root)
    loaded_count = len(candidates)
    if scope_ids:
        wanted = set(scope_ids)
        candidates = [c for c in candidates if c.scope_id in wanted]
    scope_filtered_count = len(candidates)

    # Filter candidates to only those with at least one ready dataset
    # (before tasks_written_cap, so the cap applies to actionable candidates)
    filtered_candidates = []
    skipped_details: list[tuple[str, list[str]]] = []
    for c in candidates:
        has_ready_dataset, missing = check_paper_datasets_ready(
            c.scope_id,
            c.datasets,
            datasets_root,
            curated_datasets_root=curated_datasets_root,
        )
        if has_ready_dataset:
            filtered_candidates.append(c)
        else:
            skipped_details.append((c.task_id, missing))
    candidates = filtered_candidates

    if skipped_details:
        detail_lines = [f"  {tid}: {','.join(ms)}" for tid, ms in skipped_details]
        log_detail(
            logger,
            f"{PHASE_TAG} skipped candidates (no listed datasets ready):\n"
            + "\n".join(detail_lines),
        )

    # Build catalog of existing installed tasks for dedup
    existing_tasks = _build_existing_tasks_catalog(tasks_root)
    logger.info(
        "%s   load existing installed tasks count=%d from=%s",
        PHASE_TAG,
        len(existing_tasks),
        _rel_path(tasks_root),
    )

    # Existing in-scope tasks consume the cap unless overwrite starts a fresh
    # pass. Enforce the remainder in-loop so skips do not consume budget.
    effective_tasks_written_cap = tasks_written_cap
    if tasks_written_cap is not None:
        existing_task_count_total = len(existing_tasks)
        if scope_ids:
            requested = {str(s).strip() for s in scope_ids if str(s).strip()}
            existing_task_count = sum(
                1
                for entry in existing_tasks
                if str(entry.get("scope_id", "")).strip() in requested
            )
        else:
            existing_task_count = existing_task_count_total
        effective_tasks_written_cap = (
            tasks_written_cap
            if overwrite
            else max(0, tasks_written_cap - existing_task_count)
        )
        logger.info(
            "%s   cap: total=%d existing_in_scope=%d (installed_total=%d) "
            "overwrite=%s scope_filter=%s budget=%d",
            PHASE_TAG,
            tasks_written_cap,
            existing_task_count,
            existing_task_count_total,
            overwrite,
            scope_ids or "ALL",
            effective_tasks_written_cap,
        )

    log_stage(
        logger,
        "%s starting %d tasks (existing=%d, candidates=%d, scope_filtered=%d, dataset_available=%d)",
        PHASE_TAG,
        len(candidates),
        len(existing_tasks),
        loaded_count,
        scope_filtered_count,
        len(filtered_candidates),
    )

    stats = {
        "tasks_implemented": 0,
        "tasks_skipped_duplicate": 0,
        "tasks_skipped_previously_failed": 0,
        "tasks_skipped_unsupported_family": 0,
        "tasks_skipped_trading_no_triage": 0,
        "tasks_skipped_written_cap": 0,
        "tasks_failed": 0,
        "tasks_failed_loader_unresolved": 0,
        "tasks_failed_dry_fire": 0,
        "tasks_failed_agent_smoke": 0,
        "tasks_smoke_validation_failed": 0,
    }

    # A staging folder without a corresponding installed task means the
    # previous attempt failed at some gate. Skip these by default so we
    # don't burn LLM credits re-running known-failed candidates; operator
    # opts back in via --retry-failed-attempts or --overwrite.
    previously_attempted_staging_folders: set[str] = set()
    if not retry_failed_attempts and not overwrite and staging_root.exists():
        previously_attempted_staging_folders = {
            entry.name for entry in staging_root.iterdir() if entry.is_dir()
        }
    if previously_attempted_staging_folders:
        logger.info(
            "%s   found %d previously-attempted staging folder(s) at %s "
            "(pass --retry-failed-attempts to retry)",
            PHASE_TAG,
            len(previously_attempted_staging_folders),
            _rel_path(staging_root),
        )

    # Identity-based collision tracker for current-run installs only.
    # Threaded into implement_task → install_task so a later sibling
    # cannot silently overwrite an earlier sibling's install dir.
    # Distinct from ``existing_tasks`` (semantic dedup, mixed-run).
    installed_task_ids: set[str] = set()

    for i, candidate in enumerate(candidates, 1):
        paper_ref = (
            f"{candidate.scope_id}/{candidate.paper_id}"
            if candidate.scope_id and candidate.paper_id
            else (candidate.paper_id or candidate.scope_id or "(unknown)")
        )
        log_stage(
            logger,
            "%s [%d/%d] task_id=%s paper=%s",
            PHASE_TAG,
            i,
            len(candidates),
            candidate.task_id,
            paper_ref,
        )

        # 0. Cap gate — Phase 2/3 parity. Once the per-scope budget is met,
        # short-circuit before the LLM dedup call so remaining candidates
        # don't burn credits on work that can't land. Placed above dedup
        # because Phase 4 has no useful reuse side-effects to preserve
        # (Phase 2/3 keep exact-name reuse running even after the cap).
        if (
            effective_tasks_written_cap is not None
            and stats["tasks_implemented"] >= effective_tasks_written_cap
        ):
            log_stage(
                logger,
                "%s   -> skip (tasks_written_cap_reached total_cap=%s budget=%d)",
                PHASE_TAG,
                tasks_written_cap,
                effective_tasks_written_cap,
            )
            stats["tasks_skipped_written_cap"] += 1
            continue

        # 1. Dedup check — bypassed when --overwrite forces full regeneration
        is_dup, matched_id, dedup_reason = can_reuse_existing_task(
            candidate,
            {} if overwrite else existing_tasks,
            llm,
        )
        if is_dup:
            log_detail(
                logger,
                "%s skip duplicate task_id=%s matched_existing=%s reason=%s",
                PHASE_TAG,
                candidate.task_id,
                matched_id,
                dedup_reason,
            )
            log_stage(
                logger,
                "%s   -> skip (duplicate matched=%s)",
                PHASE_TAG,
                matched_id,
            )
            stats["tasks_skipped_duplicate"] += 1
            continue

        # 1a. In-run task_id pre-check. Fan-out guarantees uniqueness by
        # construction; reaching here means a bug. Skip rather than let
        # install_task raise after the LLM calls have already been spent.
        if candidate.task_id in installed_task_ids:
            log_detail(
                logger,
                "%s skip in_run_duplicate task_id=%s — already installed "
                "earlier this run; candidate-load layer should have made "
                "this id unique",
                PHASE_TAG,
                candidate.task_id,
            )
            log_stage(
                logger,
                "%s   -> skip (in_run_duplicate)",
                PHASE_TAG,
            )
            stats["tasks_skipped_duplicate"] += 1
            continue

        # 1b. Previously-failed skip: no installed task + staging folder
        # present = prior attempt produced no install. Skip unless
        # --retry-failed-attempts.
        candidate_staging_folder = _staging_folder_name(
            candidate.task_id, candidate.ml_task_summary
        )
        if (
            not retry_failed_attempts
            and candidate_staging_folder in previously_attempted_staging_folders
        ):
            staging_path = staging_root / candidate_staging_folder
            log_detail(
                logger,
                "%s skip previously_failed task_id=%s staging=%s "
                "(pass --retry-failed-attempts to retry)",
                PHASE_TAG,
                candidate.task_id,
                staging_path,
            )
            log_stage(
                logger,
                "%s   -> skip (previously_failed)",
                PHASE_TAG,
            )
            stats["tasks_skipped_previously_failed"] += 1
            continue

        # 1c. Trading papers that escaped Phase 1c routing (triage didn't
        # run, or returned no_match/novel_task_required). Phase 4 cannot
        # auto-construct these — flag and skip.
        if str(candidate.task_family or "").strip().lower() in TRADING_FAMILIES:
            log_detail(
                logger,
                "%s skip trading_no_triage task_id=%s task_family=%s "
                "— trading-flavored paper has no curated_routed binding; "
                "run `python -m openfinai_pipeline triage-trading-papers` "
                "to route it to a curated trading task",
                PHASE_TAG,
                candidate.task_id,
                candidate.task_family,
            )
            log_stage(
                logger,
                "%s   -> skip (trading_no_triage family=%s)",
                PHASE_TAG,
                candidate.task_family,
            )
            stats["tasks_skipped_trading_no_triage"] += 1
            continue

        # 2. Enrich with Phase 2/3 artifacts
        phase_artifacts = enrich_candidate_from_phases(
            candidate,
            datasets_root,
            curated_datasets_root,
            rewards_root,
            generated_dir=generated_dir,
        )

        # 3. Generate a task-aware staging loader. Installation executes it to
        # materialize HDF5 artifacts, then writes the runtime loader.
        staging_folder = _staging_folder_name(
            candidate.task_id, candidate.ml_task_summary
        )
        task_staging_dir = staging_root / staging_folder
        interaction_model = _infer_interaction_model(candidate)
        dataset = phase_artifacts.datasets[0] if phase_artifacts.datasets else None
        loader_result: LoaderResult | None = None
        if dataset is not None:
            paper_context_for_loader = _read_paper_context(
                candidate, pdf_downloader=pdf_downloader
            )
            task_staging_dir.mkdir(parents=True, exist_ok=True)
            loader_result = generate_task_loader(
                candidate=candidate,
                dataset=dataset,
                llm=loader_llm,
                staging_dir=task_staging_dir,
                paper_context=paper_context_for_loader,
                generation_rounds=resolved_loader_generation_rounds,
                task_interaction_model=interaction_model,
            )
            log_detail(
                logger,
                "%s loader_gen status=%s rounds=%s",
                PHASE_TAG,
                loader_result.status,
                loader_result.rounds_log,
            )
            if loader_result.status == "unresolved":
                log_detail(
                    logger,
                    "%s skip task_id=%s loader generation unresolved notes=%s",
                    PHASE_TAG,
                    candidate.task_id,
                    loader_result.notes,
                )
                log_stage(
                    logger,
                    "%s   -> skip (loader_unresolved)",
                    PHASE_TAG,
                )
                stats["tasks_failed_loader_unresolved"] += 1
                continue

        # 4. Assemble evaluator.py deterministically from the reward bank
        artifact_plan = _build_artifact_plan(dataset, interaction_model)
        dataset_name_for_err = dataset.name if dataset is not None else ""
        # Resolve the dataset's data dir so file-existence checks can be
        # performed at assembly time (on-disk manifest path is
        # <data_dir>/../manifest.json alongside the data/ subdir).
        dataset_data_dir: Path | None = None
        if dataset is not None and dataset.source_paths:
            try:
                first_path = Path(dataset.source_paths[0]).resolve()
                # Walk up until we find a "data" parent.
                for parent in [first_path, *first_path.parents]:
                    if parent.name == "data":
                        dataset_data_dir = parent
                        break
            except Exception:  # noqa: BLE001 — best-effort
                dataset_data_dir = None
        try:
            _validate_artifact_plan_for_assembly(
                artifact_plan,
                task_id=candidate.task_id,
                dataset_name=dataset_name_for_err,
                dataset_data_dir=dataset_data_dir,
            )
            assembly_result = assemble_evaluator(
                task_id=candidate.task_id,
                task_title=candidate.title,
                rewards=candidate.rewards,
                eval_root=rewards_root,
                staging_dir=task_staging_dir,
                interaction_model=interaction_model,
                artifact_plan=artifact_plan,
                generated_dir=generated_dir,
            )
            log_stage(
                logger,
                "%s   evaluator -> resolved=%s unresolved=%s",
                PHASE_TAG,
                assembly_result.get("rewards_resolved", 0),
                assembly_result.get("rewards_unresolved", []),
            )
        except AssemblyGroundTruthError as exc:
            log_detail(
                logger,
                "%s evaluator assembly aborted task_id=%s reason=%s",
                PHASE_TAG,
                candidate.task_id,
                exc,
                level=logging.ERROR,
            )
            stats["tasks_failed_manifest_unresolved"] = (
                int(stats.get("tasks_failed_manifest_unresolved", 0)) + 1
            )
            assembly_result = {
                "success": False,
                "rewards_resolved": 0,
                "resolved_reward_names": [],
                "resolved_reward_keys": [],
                "rewards_unresolved": [r.get("name", "") for r in candidate.rewards],
                "manifest_error": str(exc),
            }
            log_stage(
                logger,
                "%s   evaluator -> failed: %s",
                PHASE_TAG,
                truncate_oneline(str(exc)),
            )
        except Exception as exc:
            log_detail(
                logger,
                "%s evaluator assembly failed task_id=%s error=%s",
                PHASE_TAG,
                candidate.task_id,
                exc,
                level=logging.WARNING,
            )
            assembly_result = {
                "success": False,
                "rewards_resolved": 0,
                "resolved_reward_names": [],
                "resolved_reward_keys": [],
                "rewards_unresolved": [r.get("name", "") for r in candidate.rewards],
            }
            log_stage(
                logger,
                "%s   evaluator -> failed: %s",
                PHASE_TAG,
                truncate_oneline(str(exc)),
            )

        if int(assembly_result.get("rewards_resolved", 0)) <= 0:
            log_detail(
                logger,
                "%s skip task_id=%s no_rewards_resolved unresolved=%s",
                PHASE_TAG,
                candidate.task_id,
                assembly_result.get("rewards_unresolved", []),
            )
            log_stage(
                logger,
                "%s   -> skip (no_rewards_resolved unresolved=%s)",
                PHASE_TAG,
                assembly_result.get("rewards_unresolved", []),
            )
            stats["tasks_failed"] += 1
            continue

            # Dry-fire evaluator, loader, rewards, and real ground truth before
            # task codegen. Sensitivity checks use slugified reward keys.
        resolved_reward_keys_for_gate = list(
            assembly_result.get("resolved_reward_keys", [])
        )
        if (
            dataset_data_dir is not None
            and dataset_data_dir.exists()
            and (task_staging_dir / "evaluator.py").exists()
            and (task_staging_dir / "load.py").exists()
        ):
            try:
                dry_fire_result = dry_fire_evaluator(
                    staging_dir=task_staging_dir,
                    data_dir=dataset_data_dir,
                    resolved_reward_names=resolved_reward_keys_for_gate,
                )
            except Exception as exc:  # noqa: BLE001 — defensive, never crash run
                dry_fire_result = {
                    "ran": False,
                    "ok": False,
                    "errors": [f"dry_fire_raised:{type(exc).__name__}:{exc}"],
                    "evaluator_path": (task_staging_dir / "evaluator.py").as_posix(),
                    "load_path": (task_staging_dir / "load.py").as_posix(),
                    "data_dir": dataset_data_dir.as_posix(),
                    "dry_fire_dir": (task_staging_dir / ".dry_fire").as_posix(),
                    "reward_path": "",
                    "scores": {},
                    "score_keys": [],
                    "prediction_blind": [],
                    "perfect_scores": {},
                }

            # Prune-and-retry on prediction-blind metrics: rewards that
            # produced identical finite scores for score(zeros, gt) and
            # score(gt, gt) ignore predictions on this loader's actual
            # output shape. Re-assemble with offending classes blocked
            # and re-run the gate ONCE; on no improvement or empty
            # survivor set, fall through to skip.
            blind_metrics = [
                str(k) for k in dry_fire_result.get("prediction_blind", []) if k
            ]
            if blind_metrics:
                surviving = [
                    k
                    for k in resolved_reward_keys_for_gate
                    if k not in blind_metrics
                ]
                if not surviving:
                    log_detail(
                        logger,
                        "%s task_id=%s no_prediction_sensitive_metric blind=%s; "
                        "cannot prune (no metrics survive)",
                        PHASE_TAG,
                        candidate.task_id,
                        sorted(blind_metrics),
                    )
                else:
                    log_stage(
                        logger,
                        "%s   dry-fire -> rewards_pruned: %s; re-assembling",
                        PHASE_TAG,
                        sorted(blind_metrics),
                    )
                    reassembled = False
                    try:
                        assembly_result = assemble_evaluator(
                            task_id=candidate.task_id,
                            task_title=candidate.title,
                            rewards=candidate.rewards,
                            eval_root=rewards_root,
                            staging_dir=task_staging_dir,
                            interaction_model=interaction_model,
                            artifact_plan=artifact_plan,
                            generated_dir=generated_dir,
                            block_reward_keys=blind_metrics,
                        )
                        reassembled = True
                    except Exception as exc:  # noqa: BLE001 — defensive
                        log_detail(
                            logger,
                            "%s evaluator re-assembly after prune failed "
                            "task_id=%s error=%s",
                            PHASE_TAG,
                            candidate.task_id,
                            exc,
                            level=logging.WARNING,
                        )
                    if reassembled and int(
                        assembly_result.get("rewards_resolved", 0)
                    ) > 0:
                        resolved_reward_keys_for_gate = list(
                            assembly_result.get("resolved_reward_keys", [])
                        )
                        log_detail(
                            logger,
                            "%s evaluator -> resolved=%s unresolved=%s pruned=%s",
                            PHASE_TAG,
                            assembly_result.get("rewards_resolved", 0),
                            assembly_result.get("rewards_unresolved", []),
                            assembly_result.get("rewards_pruned", []),
                        )
                        try:
                            dry_fire_result = dry_fire_evaluator(
                                staging_dir=task_staging_dir,
                                data_dir=dataset_data_dir,
                                resolved_reward_names=resolved_reward_keys_for_gate,
                            )
                        except Exception as exc:  # noqa: BLE001 — defensive
                            dry_fire_result = {
                                "ran": False,
                                "ok": False,
                                "errors": [
                                    f"dry_fire_raised:{type(exc).__name__}:{exc}"
                                ],
                                "evaluator_path": (
                                    task_staging_dir / "evaluator.py"
                                ).as_posix(),
                                "load_path": (
                                    task_staging_dir / "load.py"
                                ).as_posix(),
                                "data_dir": dataset_data_dir.as_posix(),
                                "dry_fire_dir": (
                                    task_staging_dir / ".dry_fire"
                                ).as_posix(),
                                "reward_path": "",
                                "scores": {},
                                "score_keys": [],
                                "prediction_blind": [],
                                "perfect_scores": {},
                            }

            if not dry_fire_result.get("ok", False):
                stats["tasks_failed_dry_fire"] += 1
                errors_list = list(dry_fire_result.get("errors", []))
                try:
                    sidecar_path = write_dry_fire_failure_sidecar(
                        task_staging_dir,
                        scope_id=candidate.scope_id,
                        task_id=candidate.task_id,
                        result=dry_fire_result,
                    )
                except Exception as exc:  # noqa: BLE001 — sidecar best-effort
                    sidecar_path = None
                    log_detail(
                        logger,
                        "%s dry-fire sidecar write failed task_id=%s error=%s",
                        PHASE_TAG,
                        candidate.task_id,
                        exc,
                        level=logging.WARNING,
                    )
                log_detail(
                    logger,
                    "%s skip task_id=%s dry-fire gate failed errors=%s sidecar=%s",
                    PHASE_TAG,
                    candidate.task_id,
                    errors_list,
                    sidecar_path,
                )
                log_stage(
                    logger,
                    "%s   dry-fire -> failed: %s",
                    PHASE_TAG,
                    _short_dry_fire_reason(errors_list),
                )
                log_stage(
                    logger,
                    "%s   -> skip (dry_fire_failed)",
                    PHASE_TAG,
                )
                continue
            log_stage(
                logger,
                "%s   dry-fire -> ok",
                PHASE_TAG,
            )
        else:
            log_detail(
                logger,
                "%s skip dry-fire task_id=%s: missing dry-fire prerequisites "
                "(staging_evaluator=%s staging_load=%s data_dir=%s)",
                PHASE_TAG,
                candidate.task_id,
                (task_staging_dir / "evaluator.py").exists(),
                (task_staging_dir / "load.py").exists(),
                dataset_data_dir.as_posix() if dataset_data_dir is not None else None,
                level=logging.WARNING,
            )

        # 5. Generate task.py and prose, then stage deterministic contracts.
        installed_task_dir = tasks_root / candidate.scope_id / candidate.task_id
        _update_paper_task_record(
            candidate.task_dir,
            installed_task_dir=installed_task_dir,
            generated=False,
        )
        task_log_path = task_staging_dir / "construction_log.jsonl"
        # status="ready" means the install step runs slicer + writes HDF5
        # + replaces load.py with the trivial runtime loader. Otherwise
        # (e.g. trading short-circuits to "not_required") falls back to
        # the legacy raw-payload install path.
        loader_ready = bool(
            loader_result is not None and loader_result.status == "ready"
        )
        # The instruction.md renderer needs to know whether the task ships
        # a held-out test target. Slicer hasn't run yet, but slice_b_shape
        # enforces shape↔split_policy consistency, so the loader's policy
        # is a faithful proxy. Default True when no loader / short-circuit.
        has_held_out_test_gt = (
            loader_result is None
            or (loader_result.split_policy or "") != "no_split"
        )
        # Forward the loader's ground_truth_provenance into the manifest
        # so the audit trail (source mechanism, column/artifact, transform,
        # horizon, paper quote, leakage audit) is durable post-install.
        loader_provenance = (
            loader_result.ground_truth_provenance
            if loader_result is not None
            else None
        )
        result = implement_task(
            candidate,
            llm,
            sandbox,
            staging_root,
            tasks_root,
            datasets_root,
            curated_datasets_root,
            log_path=task_log_path,
            task_generation_rounds=resolved_task_generation_rounds,
            phase_artifacts=phase_artifacts,
            resolved_reward_names=list(assembly_result.get("resolved_reward_names", [])),
            pdf_downloader=pdf_downloader,
            loader_ready=loader_ready,
            has_held_out_test_gt=has_held_out_test_gt,
            ground_truth_provenance=loader_provenance,
            installed_task_ids=installed_task_ids,
        )

        if result.success:
            _update_paper_task_record(
                candidate.task_dir,
                installed_task_dir=installed_task_dir,
                generated=True,
            )
            stats["tasks_implemented"] += 1
            installed_task_ids.add(candidate.task_id)
            # Update catalog for dedup of subsequent candidates
            existing_tasks.append(
                {
                    "task_id": candidate.task_id,
                    "title": candidate.title,
                    "description": candidate.ml_task_summary[:500],
                }
            )
            # Non-blocking smoke gate: layout pre-check + zero-prediction
            # smoke. Failures sidecar to validation_failed.json + bump
            # tasks_smoke_validation_failed; do NOT undo the install.
            validation = {"valid": False, "errors": [], "smoke": {}}
            try:
                validation = validate_installed_task_bundle(
                    installed_task_dir,
                    scope_id=candidate.scope_id,
                    task_id=candidate.task_id,
                    staging_dir=task_staging_dir,
                )
                if not validation.get("valid", False):
                    stats["tasks_smoke_validation_failed"] += 1
                    sidecar = write_validation_failure_sidecar(
                        installed_task_dir,
                        tasks_root=tasks_root,
                        scope_id=candidate.scope_id,
                        task_id=candidate.task_id,
                        result=validation,
                    )
                    log_detail(
                        logger,
                        "%s smoke validation failed task_id=%s errors=%s sidecar=%s",
                        PHASE_TAG,
                        candidate.task_id,
                        validation.get("errors", []),
                        sidecar,
                        level=logging.WARNING,
                    )
            except Exception as exc:  # noqa: BLE001 — defensive
                stats["tasks_smoke_validation_failed"] += 1
                log_detail(
                    logger,
                    "%s smoke gate raised task_id=%s error=%s (sidecar skipped)",
                    PHASE_TAG,
                    candidate.task_id,
                    exc,
                    level=logging.WARNING,
                )
                validation = {
                    "valid": False,
                    "errors": [f"smoke_gate_raised:{exc}"],
                    "smoke": {},
                }
            if validation.get("valid", False):
                log_stage(logger, "%s   smoke -> ok", PHASE_TAG)
            else:
                log_stage(
                    logger,
                    "%s   smoke -> failed: %s",
                    PHASE_TAG,
                    short_smoke_reason(validation),
                )
            log_stage(
                logger,
                "%s   -> done attempts=%d smoke_valid=%s",
                PHASE_TAG,
                result.attempts,
                validation.get("valid", False),
            )
        else:
            stats["tasks_failed"] += 1
            # Sub-bucket on the LAST round's failure stage so the summary
            # distinguishes "agent_smoke caught it" from "pytest never let
            # us reach smoke". None covers short-circuit paths.
            if getattr(result, "failure_stage", None) == "agent_smoke":
                stats["tasks_failed_agent_smoke"] += 1
            failure_stage = getattr(result, "failure_stage", None) or "n/a"
            failure_reason = ""
            error_log = getattr(result, "error_log", None) or []
            if error_log:
                failure_reason = truncate_oneline(str(error_log[0]))
            if failure_reason:
                log_stage(
                    logger,
                    "%s   -> failed attempts=%d stage=%s reason=%s",
                    PHASE_TAG,
                    result.attempts,
                    failure_stage,
                    failure_reason,
                )
            else:
                log_stage(
                    logger,
                    "%s   -> failed attempts=%d stage=%s",
                    PHASE_TAG,
                    result.attempts,
                    failure_stage,
                )

    log_stage(
        logger,
        "%s complete implemented=%d duplicate=%d previously_failed=%d "
        "trading_no_triage=%d written_cap_skipped=%d failed=%d dry_fire_failed=%d "
        "agent_smoke_failed=%d smoke_failed=%d install=%s",
        PHASE_TAG,
        stats["tasks_implemented"],
        stats["tasks_skipped_duplicate"],
        stats["tasks_skipped_previously_failed"],
        stats["tasks_skipped_trading_no_triage"],
        stats["tasks_skipped_written_cap"],
        stats["tasks_failed"],
        stats["tasks_failed_dry_fire"],
        stats["tasks_failed_agent_smoke"],
        stats["tasks_smoke_validation_failed"],
        _rel_path(tasks_root),
    )
    return stats


def _build_existing_tasks_catalog(tasks_root: Path) -> list[dict[str, str]]:
    """
    Scan benchmark bundles for existing installed task packages.
    Returns a list of {task_id, scope_id, title, description} for dedup
    and scope-aware cap accounting. The on-disk layout is
    ``tasks_root/<scope_id>/<task_id>/`` so ``scope_id`` is the parent dir.
    """
    catalog: list[dict[str, str]] = []
    if not tasks_root.exists():
        return catalog

    for scope_dir in sorted([p for p in tasks_root.iterdir() if p.is_dir()]):
        for item in sorted([p for p in scope_dir.iterdir() if p.is_dir()]):
            if item.name.startswith("_") or item.name == "__pycache__":
                continue
            task_script = item / "environment" / "data" / "task.py"
            if not (item / "task.py").exists() and not task_script.exists():
                continue
            catalog.append(
                {
                    "task_id": item.name,
                    "scope_id": scope_dir.name,
                    "title": item.name.replace("_", " ").title(),
                    "description": "",
                }
            )

    return catalog
