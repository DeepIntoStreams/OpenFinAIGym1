import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openfinai_pipeline.corpus.aggregator import (
    aggregate_datasets,
    aggregate_rewards,
    can_reuse_existing_dataset,
    can_reuse_existing_reward,
    load_task_summaries,
)
from openfinai_pipeline.corpus.catalog import (
    GENERATED_DATASET_CATALOG_NAME,
    detect_dataset_download_script_path,
    detect_dataset_payload_paths,
    find_generated_dataset_dir,
    load_dataset_catalog,
    load_reward_catalog,
    scope_match,
    slugify,
    upsert_dataset_catalog_entry,
    upsert_reward_catalog_entry,
)
from openfinai_pipeline.corpus.dataset_filter import (
    filter_task_summaries_by_dataset_availability,
)
from openfinai_pipeline.corpus.executor import (
    _REWARD_TEST_FILE,
    write_dataset_asset,
    write_reward_asset,
)
from openfinai_pipeline.llm import LLMService
from openfinai_pipeline.papers.summary_payload import extract_summary_payload
from openfinai_pipeline.settings import (
    AppConfig,
    load_app_config,
    resolve_phase_llm,
)
from openfinai_pipeline.utils.logging import (
    configure_practice_logging,
    format_paper_refs,
    log_detail,
    log_stage,
    truncate_oneline,
)

logger = logging.getLogger(__name__)
PHASE2_TAG = "[phase2:dataset]"
PHASE3_TAG = "[phase3:rewards]"


def _paper_json_path(research_root: Path, task_dir: str) -> Path:
    return research_root / task_dir / "paper.json"


def _ordered_dataset_summary_entry(
    entry: dict[str, Any],
    *,
    reused: bool,
    download_attempted: bool,
    download_successful: bool,
    downloaded_dataset_description: str,
    download_script_path: str,
    dataset_paths: list[str],
    preview: list[dict[str, Any]],
    interaction_model: str = "",
    has_ground_truth: bool = False,
    artifacts: list[dict[str, Any]] | None = None,
    manifest_status: str = "",
) -> dict[str, Any]:
    return {
        "name": str(entry.get("name", "")).strip(),
        "description": str(entry.get("description", "")).strip(),
        "dataset_kind": str(entry.get("dataset_kind", "real")).strip() or "real",
        "reused": bool(reused),
        "download_attempted": bool(download_attempted),
        "download_successful": bool(download_successful),
        "downloaded_dataset_description": str(downloaded_dataset_description).strip(),
        "download_script_path": str(download_script_path).strip(),
        "dataset_paths": [str(path).strip() for path in dataset_paths if str(path).strip()],
        "preview": [item for item in preview if isinstance(item, dict)],
        "interaction_model": str(interaction_model or "").strip(),
        "has_ground_truth": bool(has_ground_truth),
        "artifacts": [item for item in (artifacts or []) if isinstance(item, dict)],
        "manifest_status": str(manifest_status or "").strip(),
    }


def _update_paper_dataset_entries(
    *,
    research_root: Path | None,
    task_dirs: list[str],
    dataset_name: str,
    reused: bool,
    download_attempted: bool,
    download_successful: bool,
    downloaded_dataset_description: str,
    download_script_path: str,
    dataset_paths: list[str],
    preview: list[dict[str, Any]],
    interaction_model: str = "",
    has_ground_truth: bool = False,
    artifacts: list[dict[str, Any]] | None = None,
    manifest_status: str = "",
) -> None:
    if research_root is None:
        return
    for task_dir in task_dirs:
        paper_json_path = _paper_json_path(research_root, task_dir)
        if not paper_json_path.exists():
            continue
        try:
            paper_doc = json.loads(paper_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summary = extract_summary_payload(paper_doc)
        if not isinstance(summary, dict):
            continue
        datasets = summary.get("datasets")
        if not isinstance(datasets, list):
            continue
        changed = False
        for idx, item in enumerate(datasets):
            if not isinstance(item, dict):
                continue
            if str(item.get("name", "")).strip() != dataset_name:
                continue
            datasets[idx] = _ordered_dataset_summary_entry(
                item,
                reused=reused,
                download_attempted=download_attempted,
                download_successful=download_successful,
                downloaded_dataset_description=downloaded_dataset_description,
                download_script_path=download_script_path,
                dataset_paths=dataset_paths,
                preview=preview,
                interaction_model=interaction_model,
                has_ground_truth=has_ground_truth,
                artifacts=artifacts,
                manifest_status=manifest_status,
            )
            changed = True
        if changed:
            paper_doc["summary"] = summary
            paper_json_path.write_text(
                json.dumps(paper_doc, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


def _read_manifest_fields(dataset_dir: Path | None) -> dict[str, Any]:
    """Read the semantic fields from a dataset's on-disk manifest.json.

    Returns an empty-default dict when the manifest is missing or can't
    be parsed. Callers thread the returned fields into paper.json
    datasets[] entries so Phase 4 can consume them without a second
    disk read.
    """
    empty = {
        "interaction_model": "",
        "has_ground_truth": False,
        "artifacts": [],
        "manifest_status": "",
    }
    if dataset_dir is None:
        return empty
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return empty
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty
    if not isinstance(data, dict):
        return empty
    artifacts_raw = data.get("artifacts")
    if not isinstance(artifacts_raw, list):
        artifacts_raw = []
    artifacts = [item for item in artifacts_raw if isinstance(item, dict)]
    return {
        "interaction_model": str(data.get("interaction_model", "")).strip(),
        "has_ground_truth": bool(data.get("has_ground_truth", False)),
        "artifacts": artifacts,
        "manifest_status": str(data.get("manifest_status", "")).strip(),
    }


def _spec_from_catalog_entry(matched_entry: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract unified-interface fields from a ``reward_bank.json``/generated-catalog entry.

    Returns ``None`` if the entry carries no spec fields (e.g. legacy
    generated-catalog rows pre-dating the unified interface).
    """
    if not isinstance(matched_entry, dict):
        return None
    spec: dict[str, Any] = {}
    class_name = str(matched_entry.get("class_name", "")).strip()
    module_name = str(matched_entry.get("module", "")).strip()
    if class_name:
        spec["resolved_class_name"] = class_name
    if module_name:
        spec["resolved_module_name"] = module_name
    default_params = matched_entry.get("default_params")
    if isinstance(default_params, dict):
        spec["params"] = dict(default_params)
    return spec or None


def _spec_from_write_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract unified-interface fields from a ``write_reward_asset`` result.

    Under the canonical-name contract the writer's ``spec`` only carries
    class/module names + ``default_params`` (literal hyperparameter
    defaults). Param-name → ctx-key wiring is implicit in the source
    signatures and resolved at runtime via introspection.
    """
    if not isinstance(result, dict):
        return None
    spec_payload = result.get("spec")
    if not isinstance(spec_payload, dict):
        return None
    out: dict[str, Any] = {}
    for key, target in (
        ("class_name", "resolved_class_name"),
        ("module_name", "resolved_module_name"),
    ):
        value = spec_payload.get(key)
        if isinstance(value, str) and value.strip():
            out[target] = value.strip()
    default_params = spec_payload.get("default_params")
    if isinstance(default_params, dict):
        out["params"] = dict(default_params)
    return out or None


def _ordered_metric_summary_entry(
    entry: dict[str, Any],
    *,
    reused: bool,
    generated_successful: bool,
    reward_script_path: str,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical per-metric summary entry written into paper.json.

    ``spec`` carries ``params`` (catalog-snapshot hyperparameter defaults)
    and ``resolved_class_name`` / ``resolved_module_name``. Param-name →
    ctx-key wiring is implicit in the source signatures and resolved at
    runtime; no envelope is stored.
    """
    spec = spec or {}
    # Per-task hyperparameter override (may be supplied by Phase 3 LLM or
    # extracted from the paper summary); preserved across rewrites.
    params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
    return {
        "name": str(entry.get("name", "")).strip(),
        "description": str(entry.get("description", "")).strip(),
        "reused": bool(reused),
        "generated_successful": bool(generated_successful),
        "reward_script_path": str(reward_script_path).strip(),
        "params": dict(spec.get("params", params)),
        "resolved_class_name": str(
            spec.get("resolved_class_name", entry.get("resolved_class_name", ""))
        ).strip(),
        "resolved_module_name": str(
            spec.get("resolved_module_name", entry.get("resolved_module_name", ""))
        ).strip(),
    }


def _update_paper_reward_entries(
    *,
    research_root: Path | None,
    task_dirs: list[str],
    reward_name: str,
    reused: bool,
    generated_successful: bool,
    reward_script_path: str,
    spec: dict[str, Any] | None = None,
) -> None:
    if research_root is None:
        return
    for task_dir in task_dirs:
        paper_json_path = _paper_json_path(research_root, task_dir)
        if not paper_json_path.exists():
            continue
        try:
            paper_doc = json.loads(paper_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summary = extract_summary_payload(paper_doc)
        if not isinstance(summary, dict):
            continue
        metrics = summary.get("metrics")
        if not isinstance(metrics, list):
            continue
        changed = False
        for idx, item in enumerate(metrics):
            if not isinstance(item, dict):
                continue
            if str(item.get("name", "")).strip() != reward_name:
                continue
            metrics[idx] = _ordered_metric_summary_entry(
                item,
                reused=reused,
                generated_successful=generated_successful,
                reward_script_path=reward_script_path,
                spec=spec,
            )
            changed = True
        if changed:
            paper_doc["summary"] = summary
            paper_json_path.write_text(
                json.dumps(paper_doc, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


def _task_dirs_from_summaries(
    task_summaries: list[tuple[str, dict[str, object]]],
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for task_dir, _summary in task_summaries:
        clean = str(task_dir).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def _candidate_scope_ids(candidate: Any) -> list[str]:
    """Union of a candidate's primary scope + any supporting scopes, deduped."""
    primary = str(getattr(candidate, "scope_id", "") or "").strip()
    supporting = list(getattr(candidate, "supporting_scope_ids", []) or [])
    collected: set[str] = set()
    if primary:
        collected.add(primary)
    for s in supporting:
        cleaned = str(s).strip()
        if cleaned:
            collected.add(cleaned)
    return sorted(collected)


def _merge_dataset_scope_ids_into_catalog(
    *,
    generated_catalog: list[dict[str, Any]],
    generated_catalog_path: Path,
    dataset_name: str,
    scope_ids_to_add: list[str],
) -> bool:
    """Union *scope_ids_to_add* into the named entry's scope_ids; persist if
    the set actually grew. Returns True iff the on-disk catalog was rewritten.
    """
    if not scope_ids_to_add:
        return False
    target_name = dataset_name.strip()
    if not target_name:
        return False
    for entry in generated_catalog:
        if str(entry.get("name", "")).strip() != target_name:
            continue
        existing = {
            str(s).strip()
            for s in entry.get("scope_ids", [])
            if str(s).strip()
        }
        merged = existing | {s for s in scope_ids_to_add if s}
        if merged == existing:
            return False
        entry["scope_ids"] = sorted(merged)
        generated_catalog_path.write_text(
            json.dumps(generated_catalog, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return True
    return False


def _merge_reward_scope_ids_into_catalog(
    *,
    generated_catalog: list[dict[str, Any]],
    generated_catalog_path: Path,
    reward_name: str,
    scope_ids_to_add: list[str],
) -> bool:
    """Reward-catalog twin of _merge_dataset_scope_ids_into_catalog."""
    if not scope_ids_to_add:
        return False
    target_name = reward_name.strip()
    if not target_name:
        return False
    for entry in generated_catalog:
        if str(entry.get("name", "")).strip() != target_name:
            continue
        existing = {
            str(s).strip()
            for s in entry.get("scope_ids", [])
            if str(s).strip()
        }
        merged = existing | {s for s in scope_ids_to_add if s}
        if merged == existing:
            return False
        entry["scope_ids"] = sorted(merged)
        # source_kind is load-time only; drop before persisting.
        on_disk = [
            {k: v for k, v in e.items() if k != "source_kind"}
            for e in generated_catalog
        ]
        generated_catalog_path.write_text(
            json.dumps(on_disk, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return True
    return False


def _reset_paper_dataset_runtime_fields(
    *,
    research_root: Path | None,
    task_dirs: list[str],
) -> None:
    if research_root is None:
        return
    for task_dir in task_dirs:
        paper_json_path = _paper_json_path(research_root, task_dir)
        if not paper_json_path.exists():
            continue
        try:
            paper_doc = json.loads(paper_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summary = extract_summary_payload(paper_doc)
        if not isinstance(summary, dict):
            continue
        datasets = summary.get("datasets")
        if not isinstance(datasets, list):
            continue
        changed = False
        for idx, item in enumerate(datasets):
            if not isinstance(item, dict):
                continue
            normalized = _ordered_dataset_summary_entry(
                item,
                reused=False,
                download_attempted=False,
                download_successful=False,
                downloaded_dataset_description="",
                download_script_path="",
                dataset_paths=[],
                preview=[],
            )
            if normalized != item:
                datasets[idx] = normalized
                changed = True
        if changed:
            paper_doc["summary"] = summary
            paper_json_path.write_text(
                json.dumps(paper_doc, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


def _reset_paper_reward_runtime_fields(
    *,
    research_root: Path | None,
    task_dirs: list[str],
) -> None:
    if research_root is None:
        return
    for task_dir in task_dirs:
        paper_json_path = _paper_json_path(research_root, task_dir)
        if not paper_json_path.exists():
            continue
        try:
            paper_doc = json.loads(paper_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summary = extract_summary_payload(paper_doc)
        if not isinstance(summary, dict):
            continue
        metrics = summary.get("metrics")
        if not isinstance(metrics, list):
            continue
        changed = False
        for idx, item in enumerate(metrics):
            if not isinstance(item, dict):
                continue
            normalized = _ordered_metric_summary_entry(
                item,
                reused=False,
                generated_successful=False,
                reward_script_path="",
            )
            if normalized != item:
                metrics[idx] = normalized
                changed = True
        if changed:
            paper_doc["summary"] = summary
            paper_json_path.write_text(
                json.dumps(paper_doc, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


def _update_generated_dataset_usage_manifest(
    *,
    datasets_root: Path,
    dataset_name: str,
    task_dirs: list[str],
) -> None:
    """Merge new ``task_dirs`` into the dataset's on-disk manifest.

    Preserves all labeler-written semantic fields (role assignments,
    ``has_ground_truth``, etc.) and only mutates ``used_by_papers``.
    """
    dataset_dir = find_generated_dataset_dir(datasets_root, dataset_name)
    if dataset_dir is None:
        return
    manifest_path = dataset_dir / "manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            loaded = None
        if isinstance(loaded, dict):
            existing = dict(loaded)
    used_by = existing.get("used_by_papers", [])
    if not isinstance(used_by, list):
        used_by = []
    else:
        used_by = list(used_by)
    seen = {
        str(item.get("paper_dir", "")).strip()
        for item in used_by
        if isinstance(item, dict)
    }
    for task_dir in task_dirs:
        clean = str(task_dir).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        used_by.append({"paper_dir": clean})
    # Preserve all existing keys (labeler output, description, etc.) and
    # only overwrite name + used_by_papers.
    existing["name"] = dataset_name
    existing["used_by_papers"] = used_by
    manifest_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _resolve_reused_dataset_details(
    *,
    datasets_root: Path,
    curated_datasets_root: Path,
    matched_entry: dict[str, Any],
) -> dict[str, Any]:
    source_kind = str(matched_entry.get("source_kind", "")).strip().lower()
    preview = [
        item for item in matched_entry.get("preview", [])
        if isinstance(item, dict)
    ]
    if source_kind == "generated":
        downloaded_dataset_description = str(
            matched_entry.get("downloaded_dataset_description", "")
        ).strip() or str(matched_entry.get("description", "")).strip()
        dataset_dir = find_generated_dataset_dir(
            datasets_root,
            str(matched_entry.get("name", "")).strip(),
        )
        if dataset_dir is None:
            return {}
        payload_paths = [
            str((datasets_root / rel_path).resolve())
            for rel_path in detect_dataset_payload_paths(dataset_dir)
            if (datasets_root / rel_path).exists()
        ]
        manifest_fields = _read_manifest_fields(dataset_dir)
        return {
            "dataset_dir": str(dataset_dir),
            "download_script_path": detect_dataset_download_script_path(dataset_dir),
            "dataset_paths": payload_paths,
            "preview": preview,
            "downloaded_dataset_description": downloaded_dataset_description,
            "interaction_model": manifest_fields["interaction_model"],
            "has_ground_truth": manifest_fields["has_ground_truth"],
            "artifacts": manifest_fields["artifacts"],
            "manifest_status": manifest_fields["manifest_status"],
        }

    payload_files = [
        str(rel_path).strip()
        for rel_path in matched_entry.get("payload_files", [])
        if str(rel_path).strip()
    ]
    downloaded_dataset_description = str(
        matched_entry.get("description", "")
    ).strip()
    return {
        "dataset_dir": "",
        "download_script_path": "",
        "dataset_paths": [
            str((curated_datasets_root / rel_path).resolve())
            for rel_path in payload_files
            if (curated_datasets_root / rel_path).exists()
        ],
        "preview": preview,
        "downloaded_dataset_description": downloaded_dataset_description,
        # Curated datasets are out of scope for automated labeling (per
        # plan). Phase 4 sees this status and emits a targeted error that
        # tells the operator to author a manifest manually.
        "interaction_model": "",
        "has_ground_truth": False,
        "artifacts": [],
        "manifest_status": "unlabeled_curated",
    }


def _resolve_generated_reward_script_path(
    *,
    generated_dir: Path,
    matched_entry: dict[str, Any],
) -> str:
    reward_name = str(matched_entry.get("name", "")).strip()
    reward_slug = slugify(reward_name)
    if not reward_slug:
        return ""
    script_path = generated_dir / f"{reward_slug}.py"
    return str(script_path.resolve()) if script_path.exists() else ""


def _resolve_curated_reward_script_path(
    *,
    eval_root: Path,
    matched_entry: dict[str, Any],
) -> str:
    module_name = str(matched_entry.get("module", "")).strip()
    if not module_name:
        return ""
    script_path = eval_root / f"{module_name}.py"
    return str(script_path.resolve()) if script_path.exists() else ""


def _resolve_reused_reward_script_path(
    *,
    eval_root: Path,
    generated_dir: Path,
    matched_entry: dict[str, Any],
) -> str:
    source_kind = str(matched_entry.get("source_kind", "")).strip().lower()
    if source_kind == "generated":
        return _resolve_generated_reward_script_path(
            generated_dir=generated_dir,
            matched_entry=matched_entry,
        )
    if source_kind == "curated":
        return _resolve_curated_reward_script_path(
            eval_root=eval_root,
            matched_entry=matched_entry,
        )
    if str(matched_entry.get("module", "")).strip():
        return _resolve_curated_reward_script_path(
            eval_root=eval_root,
            matched_entry=matched_entry,
        )
    return _resolve_generated_reward_script_path(
        generated_dir=generated_dir,
        matched_entry=matched_entry,
    )


def construct_dataset(
    config_path: str | None = None,
    *,
    datasets_written_cap: int | None = None,
    scope_ids: list[str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    llm_max_retries: int | None = None,
    timeout_sec: float | None = None,
    tool_max_turns: int | None = None,
    dataset_generation_rounds: int | None = None,
    overwrite: bool = False,
    retry_failed_attempts: bool = False,
    verbose: bool = False,
) -> dict[str, int]:
    cfg = load_app_config(config_path)
    llm_cfg = resolve_phase_llm(
        cfg.llm,
        "dataset",
        cli_provider=provider,
        cli_model=model,
        cli_temperature=temperature,
        cli_max_tokens=max_tokens,
        cli_llm_max_retries=llm_max_retries,
        cli_timeout_sec=timeout_sec,
        cli_tool_max_turns=tool_max_turns,
    )
    dataset_generation_rounds = _resolve_dataset_generation_rounds(
        cfg, dataset_generation_rounds
    )
    run_tag = f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}_construct_dataset"
    run_dir = configure_practice_logging(cfg.logging.dir, "construct_dataset", run_tag=run_tag, verbose=verbose)
    llm = LLMService(llm_cfg)
    task_summaries = load_task_summaries(Path(cfg.papers.root_dir))
    loaded_count = len(task_summaries)
    task_summaries = _filter_task_summaries(task_summaries, scope_ids)
    task_summaries, papers_skipped_trading = _drop_trading_papers(task_summaries)
    log_stage(
        logger,
        "%s start papers=%d (loaded=%d skipped_trading=%d) scope_filter=%s cap=%s provider=%s model=%s",
        PHASE2_TAG,
        len(task_summaries),
        loaded_count,
        papers_skipped_trading,
        scope_ids or "ALL",
        datasets_written_cap if datasets_written_cap is not None else "None",
        llm_cfg.provider,
        llm_cfg.providers[llm_cfg.provider].model,
    )
    logger.info(
        "%s start detail config=%s papers_root=%s datasets_root=%s logs_dir=%s dataset_generation_rounds=%d overwrite=%s retry_failed_attempts=%s",
        PHASE2_TAG,
        config_path or "default",
        cfg.papers.root_dir,
        cfg.datasets.root_dir,
        run_dir,
        dataset_generation_rounds,
        overwrite,
        retry_failed_attempts,
    )
    candidates = _write_dataset_assets(
        research_root=Path(cfg.papers.root_dir),
        task_summaries=task_summaries,
        datasets_root=Path(cfg.datasets.root_dir),
        curated_datasets_root=Path(cfg.datasets.curated_root_dir),
        llm=llm,
        timeout_sec=cfg.benchmark.download_timeout_sec,
        execute_download=cfg.benchmark.execute_download,
        written_cap=datasets_written_cap,
        summary_dir=run_dir,
        staging_root=Path(cfg.staging.root_dir),
        dataset_generation_rounds=dataset_generation_rounds,
        overwrite=overwrite,
        retry_failed_attempts=retry_failed_attempts,
        scope_ids=scope_ids,
    )
    log_stage(
        logger,
        "%s complete datasets_written=%d datasets_reused=%d raw_mentions=%d candidates=%d skipped_written_cap=%d skipped_previously_failed=%d output_root=%s",
        PHASE2_TAG,
        len(candidates["written"]),
        int(candidates.get("reused", 0)),
        int(candidates.get("raw_mentions_count", 0)),
        int(candidates.get("candidate_count", 0)),
        int(candidates.get("skipped_written_cap", 0)),
        int(candidates.get("skipped_previously_failed", 0)),
        cfg.datasets.root_dir,
    )
    return {
        "datasets_written": len(candidates["written"]),
        "raw_mentions": int(candidates.get("raw_mentions_count", 0)),
        "candidates": int(candidates.get("candidate_count", 0)),
        "datasets_reused": int(candidates.get("reused", 0)),
        "datasets_skipped_written_cap": int(candidates.get("skipped_written_cap", 0)),
        "datasets_skipped_previously_failed": int(
            candidates.get("skipped_previously_failed", 0)
        ),
        "papers_skipped_trading": papers_skipped_trading,
    }


def construct_rewards(
    config_path: str | None = None,
    *,
    rewards_written_cap: int | None = None,
    scope_ids: list[str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    llm_max_retries: int | None = None,
    timeout_sec: float | None = None,
    tool_max_turns: int | None = None,
    reward_generation_rounds: int | None = None,
    overwrite: bool = False,
    retry_failed_attempts: bool = False,
    verbose: bool = False,
) -> dict[str, int]:
    cfg = load_app_config(config_path)
    llm_cfg = resolve_phase_llm(
        cfg.llm,
        "reward",
        cli_provider=provider,
        cli_model=model,
        cli_temperature=temperature,
        cli_max_tokens=max_tokens,
        cli_llm_max_retries=llm_max_retries,
        cli_timeout_sec=timeout_sec,
        cli_tool_max_turns=tool_max_turns,
    )
    reward_generation_rounds = _resolve_reward_generation_rounds(
        cfg, reward_generation_rounds
    )
    run_tag = f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}_construct_rewards"
    run_dir = configure_practice_logging(cfg.logging.dir, "construct_rewards", run_tag=run_tag, verbose=verbose)
    llm = LLMService(llm_cfg)
    task_summaries = load_task_summaries(Path(cfg.papers.root_dir))
    loaded_count = len(task_summaries)
    task_summaries = _filter_task_summaries(task_summaries, scope_ids)
    task_summaries, papers_skipped_trading = _drop_trading_papers(task_summaries)
    task_summaries = filter_task_summaries_by_dataset_availability(
        task_summaries,
        Path(cfg.datasets.root_dir),
        curated_datasets_root=Path(cfg.datasets.curated_root_dir),
    )
    log_stage(
        logger,
        "%s start papers=%d (loaded=%d skipped_trading=%d) scope_filter=%s cap=%s provider=%s model=%s",
        PHASE3_TAG,
        len(task_summaries),
        loaded_count,
        papers_skipped_trading,
        scope_ids or "ALL",
        rewards_written_cap if rewards_written_cap is not None else "None",
        llm_cfg.provider,
        llm_cfg.providers[llm_cfg.provider].model,
    )
    logger.info(
        "%s start detail config=%s papers_root=%s rewards_root=%s logs_dir=%s reward_generation_rounds=%d overwrite=%s retry_failed_attempts=%s",
        PHASE3_TAG,
        config_path or "default",
        cfg.papers.root_dir,
        cfg.rewards.root_dir,
        run_dir,
        reward_generation_rounds,
        overwrite,
        retry_failed_attempts,
    )
    rewards = _write_reward_assets(
        research_root=Path(cfg.papers.root_dir),
        task_summaries=task_summaries,
        eval_root=Path(cfg.rewards.root_dir),
        llm=llm,
        generated_dir=Path(cfg.rewards.generated_dir),
        staging_root=Path(cfg.staging.root_dir),
        written_cap=rewards_written_cap,
        summary_dir=run_dir,
        reward_generation_rounds=reward_generation_rounds,
        overwrite=overwrite,
        retry_failed_attempts=retry_failed_attempts,
        scope_ids=scope_ids,
    )
    log_stage(
        logger,
        "%s complete rewards_written=%d rewards_reused=%d raw_mentions=%d unique_named_candidates=%d skipped_written_cap=%d skipped_previously_failed=%d output_root=%s",
        PHASE3_TAG,
        len(rewards["written"]),
        int(rewards.get("reused", 0)),
        int(rewards.get("raw_mentions_count", 0)),
        int(rewards.get("candidate_count", 0)),
        int(rewards.get("skipped_written_cap", 0)),
        int(rewards.get("skipped_previously_failed", 0)),
        cfg.rewards.root_dir,
    )
    return {
        "rewards_written": len(rewards["written"]),
        "raw_mentions": int(rewards.get("raw_mentions_count", 0)),
        "unique_named_candidates": int(rewards.get("candidate_count", 0)),
        "rewards_reused": int(rewards.get("reused", 0)),
        "rewards_skipped_written_cap": int(rewards.get("skipped_written_cap", 0)),
        "rewards_skipped_previously_failed": int(
            rewards.get("skipped_previously_failed", 0)
        ),
        "papers_skipped_trading": papers_skipped_trading,
    }


def _write_dataset_assets(
    *,
    research_root: Path | None = None,
    task_summaries: list[tuple[str, dict[str, object]]],
    datasets_root: Path,
    curated_datasets_root: Path,
    llm: LLMService | None,
    timeout_sec: int,
    execute_download: bool,
    written_cap: int | None = None,
    summary_dir: Path | None = None,
    staging_root: Path | None = None,
    dataset_generation_rounds: int = 2,
    overwrite: bool = False,
    retry_failed_attempts: bool = False,
    scope_ids: list[str] | None = None,
) -> dict[str, object]:
    datasets_root.mkdir(parents=True, exist_ok=True)
    if staging_root is None:
        staging_root = datasets_root.parent / ".staging"
    generated_catalog_path = datasets_root / GENERATED_DATASET_CATALOG_NAME
    generated_catalog = load_dataset_catalog(generated_catalog_path)
    curated_catalog_path = curated_datasets_root / "datasets.json"
    curated_catalog = load_dataset_catalog(curated_catalog_path)
    reuse_catalog = [
        dict(entry, source_kind="curated") for entry in curated_catalog
    ] + [
        dict(entry, source_kind="generated") for entry in generated_catalog
    ]
    logger.info(
        "%s load dataset catalogs curated_entries=%d generated_entries=%d curated_from=%s generated_from=%s",
        PHASE2_TAG,
        len(curated_catalog),
        len(generated_catalog),
        curated_catalog_path,
        generated_catalog_path,
    )
    # Staging directories without catalog entries mark failed prior attempts.
    # Keep exact-name reuse, but skip repeated fuzzy matching and generation
    # unless retry_failed_attempts or overwrite requests a fresh pass.
    previously_attempted_dataset_names: set[str] = set()
    staging_datasets_dir = staging_root / "datasets"
    if not overwrite and staging_datasets_dir.exists():
        previously_attempted_dataset_names = {
            entry.name for entry in staging_datasets_dir.iterdir() if entry.is_dir()
        }
    skip_previously_failed = (
        not overwrite and not retry_failed_attempts
    )
    if previously_attempted_dataset_names:
        logger.info(
            "%s found %d previously-attempted dataset name(s) in staging=%s; "
            "their LLM fuzzy reuse check will be skipped on this run "
            "(skip_previously_failed=%s; pass --retry-failed-attempts or "
            "--overwrite to retry)",
            PHASE2_TAG,
            len(previously_attempted_dataset_names),
            staging_datasets_dir,
            skip_previously_failed,
        )
    # Compute effective cap: absolute total minus existing entries that
    # already belong to the requested scope(s). Datasets tagged with other
    # scopes do not consume this scope's budget; untagged legacy entries
    # are treated as scope-universal and still count (see scope_match).
    effective_cap = written_cap
    if written_cap is not None:
        existing_count_total = len(generated_catalog)
        existing_count = sum(
            1 for entry in generated_catalog if scope_match(entry, scope_ids)
        )
        effective_cap = max(0, written_cap - existing_count)
        logger.info(
            "%s absolute cap: total_cap=%d existing_in_scope=%d (catalog_total=%d) "
            "scope_filter=%s remaining_budget=%d",
            PHASE2_TAG,
            written_cap,
            existing_count,
            existing_count_total,
            scope_ids or "ALL",
            effective_cap,
        )
        if effective_cap <= 0:
            logger.info(
                "%s cap already met for new dataset writes total_cap=%d existing_in_scope=%d; continuing in reuse-only mode",
                PHASE2_TAG,
                written_cap,
                existing_count,
            )
    aggregation = aggregate_datasets(
        task_summaries,
        existing_catalog=reuse_catalog,
        llm=llm,
        debug_dir=summary_dir,
    )
    candidates = aggregation.candidates
    if overwrite:
        _reset_paper_dataset_runtime_fields(
            research_root=research_root,
            task_dirs=_task_dirs_from_summaries(task_summaries),
        )
    logger.info(
        "%s aggregate dataset candidates raw_mentions=%d candidates=%d from_paper_summaries=%d into=%s "
        "(note: dataset candidates are NOT name-clustered here; each raw mention becomes its own "
        "candidate, so candidates == raw_mentions; reuse against the catalog is decided per-candidate later)",
        PHASE2_TAG,
        aggregation.raw_mentions_count,
        aggregation.candidate_count,
        len(task_summaries),
        datasets_root,
    )
    summary_rows: list[dict[str, object]] = []
    written: list[object] = []
    reused_count = 0
    skipped_written_cap_count = 0
    skipped_previously_failed_count = 0
    insufficient_evidence_count = 0
    failed_execution_count = 0
    not_approved_count = 0

    log_stage(
        logger,
        "%s processing %d dataset candidates",
        PHASE2_TAG,
        len(candidates),
    )
    for candidate_index, candidate in enumerate(candidates, start=1):
        log_stage(
            logger,
            "%s [%d/%d] candidate=%s kind=%s papers=%s",
            PHASE2_TAG,
            candidate_index,
            len(candidates),
            candidate.name,
            candidate.dataset_kind,
            format_paper_refs(candidate.task_dirs) or "(none)",
        )
        # Once capped or previously attempted, retain free exact-name reuse but
        # suppress the fuzzy LLM call by passing llm=None.
        reuse_only_mode = effective_cap is not None and len(written) >= effective_cap
        already_attempted = (
            candidate.directory_name in previously_attempted_dataset_names
        )
        skip_llm_fuzzy_match = reuse_only_mode or already_attempted
        if already_attempted and not reuse_only_mode:
            log_detail(
                logger,
                "%s   -> skip_llm_fuzzy_match candidate=%s reason=previously_attempted "
                "(staging dir at %s)",
                PHASE2_TAG,
                candidate.name,
                staging_datasets_dir / candidate.directory_name,
            )
        reused, matched_entry, reason = can_reuse_existing_dataset(
            candidate, reuse_catalog, None if skip_llm_fuzzy_match else llm
        )
        if reused:
            reused_count += 1
            matched_name = str(matched_entry.get("name", "")).strip() if matched_entry else ""
            matched_source_kind = str(matched_entry.get("source_kind", "")).strip() if matched_entry else ""
            # Tag the matched generated-catalog entry with this candidate's
            # scope so a future cap query for the same scope correctly counts
            # this dataset as already-claimed. Curated matches are scope-
            # universal and need no tagging.
            if matched_source_kind == "generated" and matched_name:
                _merge_dataset_scope_ids_into_catalog(
                    generated_catalog=generated_catalog,
                    generated_catalog_path=generated_catalog_path,
                    dataset_name=matched_name,
                    scope_ids_to_add=_candidate_scope_ids(candidate),
                )
            reuse_details = (
                _resolve_reused_dataset_details(
                    datasets_root=datasets_root,
                    curated_datasets_root=curated_datasets_root,
                    matched_entry=matched_entry,
                )
                if matched_entry is not None
                else {}
            )
            # Per-task ``load.py`` is generated by Phase 4 against the full
            # task context (paper + summary + rewards), so Phase 2 reuse
            # only needs to propagate the labeler's manifest fields.
            _update_paper_dataset_entries(
                research_root=research_root,
                task_dirs=candidate.task_dirs,
                dataset_name=candidate.name,
                reused=True,
                download_attempted=False,
                download_successful=bool(reuse_details.get("dataset_paths", [])),
                downloaded_dataset_description=str(
                    reuse_details.get("downloaded_dataset_description", "")
                ),
                download_script_path=str(reuse_details.get("download_script_path", "")),
                dataset_paths=list(reuse_details.get("dataset_paths", [])),
                preview=list(reuse_details.get("preview", [])),
                interaction_model=str(reuse_details.get("interaction_model", "")),
                has_ground_truth=bool(reuse_details.get("has_ground_truth", False)),
                artifacts=list(reuse_details.get("artifacts", [])),
                manifest_status=str(reuse_details.get("manifest_status", "")),
            )
            if matched_source_kind == "generated":
                _update_generated_dataset_usage_manifest(
                    datasets_root=datasets_root,
                    dataset_name=matched_name,
                    task_dirs=candidate.task_dirs,
                )
            log_stage(
                logger,
                "%s   -> reused (matched=%s source=%s reason=%s resolved_paths=%d)",
                PHASE2_TAG,
                matched_name,
                matched_source_kind,
                truncate_oneline(reason),
                len(list(reuse_details.get("dataset_paths", []))),
            )
            summary_rows.append(
                {
                    "scope_id": candidate.scope_id,
                    "supporting_scope_ids": candidate.supporting_scope_ids,
                    "name": candidate.name,
                    "dataset_kind": candidate.dataset_kind,
                    "status": "reused_existing",
                    "matched_existing": matched_name,
                    "matched_existing_source_kind": matched_source_kind,
                    "reason": reason,
                    "final_review": reason,
                    "model": "",
                    "data_path": (
                        list(reuse_details.get("dataset_paths", []))[0]
                        if list(reuse_details.get("dataset_paths", []))
                        else ""
                    ),
                    "approved": bool(reuse_details.get("dataset_paths", [])),
                    "catalog_updated": False,
                }
            )
            continue

        # Reuse did not match this candidate. If a staging dir exists for
        # the same directory_name, the previous attempt did not materialize
        # (any of: insufficient_evidence, failed_execution, not_approved) —
        # by default we skip it on the rerun to avoid burning LLM credits
        # and a network roundtrip on a known-failed candidate. The operator
        # opts back in via --retry-failed-attempts or --overwrite.
        if (
            skip_previously_failed
            and candidate.directory_name in previously_attempted_dataset_names
        ):
            skipped_previously_failed_count += 1
            staging_path = staging_datasets_dir / candidate.directory_name
            log_stage(
                logger,
                "%s   -> skipped (previously_failed staging=%s; pass "
                "--retry-failed-attempts to retry)",
                PHASE2_TAG,
                staging_path,
            )
            summary_rows.append(
                {
                    "scope_id": candidate.scope_id,
                    "supporting_scope_ids": candidate.supporting_scope_ids,
                    "name": candidate.name,
                    "dataset_kind": candidate.dataset_kind,
                    "status": "skipped_previously_failed",
                    "matched_existing": "",
                    "reason": "previously_failed_staging_dir_present",
                    "final_review": "previously_failed_staging_dir_present",
                    "model": "",
                    "data_path": "",
                    "approved": False,
                    "catalog_updated": False,
                }
            )
            _update_paper_dataset_entries(
                research_root=research_root,
                task_dirs=candidate.task_dirs,
                dataset_name=candidate.name,
                reused=False,
                download_attempted=False,
                download_successful=False,
                downloaded_dataset_description="",
                download_script_path="",
                dataset_paths=[],
                preview=[],
            )
            continue

        if effective_cap is not None and len(written) >= effective_cap:
            skipped_written_cap_count += 1
            log_stage(
                logger,
                "%s   -> skipped (written_cap_reached total_cap=%s effective_cap=%d)",
                PHASE2_TAG,
                written_cap,
                effective_cap,
            )
            summary_rows.append(
                {
                    "scope_id": candidate.scope_id,
                    "supporting_scope_ids": candidate.supporting_scope_ids,
                    "name": candidate.name,
                    "dataset_kind": candidate.dataset_kind,
                    "status": "skipped_written_cap",
                    "matched_existing": "",
                    "reason": "written_cap_reached",
                    "final_review": "written_cap_reached",
                    "model": "",
                    "data_path": "",
                    "approved": False,
                    "catalog_updated": False,
                }
            )
            _update_paper_dataset_entries(
                research_root=research_root,
                task_dirs=candidate.task_dirs,
                dataset_name=candidate.name,
                reused=False,
                download_attempted=False,
                download_successful=False,
                downloaded_dataset_description="",
                download_script_path="",
                dataset_paths=[],
                preview=[],
            )
            continue

        write_result = write_dataset_asset(
            datasets_root=datasets_root,
            candidate=candidate,
            llm=llm,
            timeout_sec=timeout_sec,
            execute_download=execute_download,
            staging_root=staging_root,
            dataset_generation_rounds=dataset_generation_rounds,
        )
        final_success = bool(write_result.get("final_success", False))
        approved = bool(write_result.get("approved", False))
        materialized = final_success and approved
        generation_status = (
            str(write_result.get("generation_status", "failed_execution")).strip()
            or "failed_execution"
        )
        if materialized:
            generation_status = "materialized"
            written.append(candidate)
        elif generation_status == "insufficient_evidence":
            insufficient_evidence_count += 1
        elif generation_status == "not_approved":
            not_approved_count += 1
        else:
            failed_execution_count += 1

        downloaded_dataset_description = str(
            write_result.get("downloaded_dataset_description", "")
        ).strip()
        dataset_paths = [
            str(path).strip()
            for path in write_result.get("dataset_paths", [])
            if str(path).strip()
        ]
        preview = [
            item for item in write_result.get("preview", [])
            if isinstance(item, dict)
        ]
        download_script_path = str(write_result.get("download_script_path", "")).strip()
        # On successful materialization the full manifest was just written
        # to disk; read it back so we can thread role/format info into
        # paper.json. Per-task ``load.py`` is generated by Phase 4 — Phase
        # 2 stops at the labeled manifest.
        manifest_fields = (
            _read_manifest_fields(find_generated_dataset_dir(datasets_root, candidate.name))
            if materialized
            else {
                "interaction_model": "",
                "has_ground_truth": False,
                "artifacts": [],
                "manifest_status": "",
            }
        )
        _update_paper_dataset_entries(
            research_root=research_root,
            task_dirs=candidate.task_dirs,
            dataset_name=candidate.name,
            reused=False,
            download_attempted=bool(write_result.get("download_attempted", False)),
            download_successful=materialized,
            downloaded_dataset_description=downloaded_dataset_description,
            download_script_path=download_script_path if materialized else "",
            dataset_paths=dataset_paths if materialized else [],
            preview=preview if materialized else [],
            interaction_model=manifest_fields["interaction_model"],
            has_ground_truth=manifest_fields["has_ground_truth"],
            artifacts=manifest_fields["artifacts"],
            manifest_status=manifest_fields["manifest_status"],
        )
        if materialized:
            _update_generated_dataset_usage_manifest(
                datasets_root=datasets_root,
                dataset_name=candidate.name,
                task_dirs=candidate.task_dirs,
            )
        catalog_updated = False
        if materialized and dataset_paths:
            candidate_scopes = _candidate_scope_ids(candidate)
            catalog_updated = upsert_dataset_catalog_entry(
                generated_catalog,
                name=candidate.name,
                description=candidate.description,
                downloaded_dataset_description=downloaded_dataset_description,
                preview=preview,
                has_ground_truth=bool(manifest_fields["has_ground_truth"]),
                scope_ids=candidate_scopes,
            )
            if catalog_updated:
                generated_catalog_path.write_text(
                    json.dumps(generated_catalog, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                logger.info(
                    "%s update dataset catalog path=%s total_entries_now=%d",
                    PHASE2_TAG,
                    generated_catalog_path,
                    len(generated_catalog),
                )
        log_stage(
            logger,
            "%s   -> %s (approved=%s catalog_updated=%s payload_path=%s)",
            PHASE2_TAG,
            generation_status,
            approved,
            catalog_updated,
            dataset_paths[0] if dataset_paths else "",
        )
        summary_rows.append(
                {
                    "scope_id": candidate.scope_id,
                    "supporting_scope_ids": candidate.supporting_scope_ids,
                    "name": candidate.name,
                    "dataset_kind": candidate.dataset_kind,
                    "status": generation_status,
                    "matched_existing": "",
                    "reason": "",
                    "final_review": json.dumps(
                        write_result.get("final_review", {}).get("suggestions", []),
                    ensure_ascii=False,
                ),
                "model": str(write_result.get("model", "")),
                    "data_path": dataset_paths[0] if dataset_paths else "",
                    "approved": approved,
                    "catalog_updated": catalog_updated,
                    }
                )

    return {
        "written": written,
        "raw_mentions_count": aggregation.raw_mentions_count,
        "candidate_count": aggregation.candidate_count,
        "reused": reused_count,
        "skipped_written_cap": skipped_written_cap_count,
        "skipped_previously_failed": skipped_previously_failed_count,
    }


def _write_reward_assets(
    *,
    research_root: Path | None = None,
    task_summaries: list[tuple[str, dict[str, object]]],
    eval_root: Path,
    llm: LLMService | None,
    generated_dir: Path | None = None,
    staging_root: Path | None = None,
    written_cap: int | None = None,
    summary_dir: Path | None = None,
    reward_generation_rounds: int = 3,
    overwrite: bool = False,
    retry_failed_attempts: bool = False,
    scope_ids: list[str] | None = None,
) -> dict[str, object]:
    # Precondition: the unified reward-contract pytest file must exist.
    # If it's missing every generated candidate will be rejected by the
    # reviewer ("pytest didn't run") and the whole LLM budget burns for
    # nothing. Fail loudly here so the operator catches harness drift
    # immediately instead of after a multi-hour run.
    if not _REWARD_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Reward contract test file not found at {_REWARD_TEST_FILE}. "
            "Phase 3 reward generation requires this pytest contract to "
            "validate generated reward_fn scripts. Check that the repository "
            "is fully checked out and the path in executor.py matches the "
            "current test layout."
        )
    eval_root.mkdir(parents=True, exist_ok=True)
    if staging_root is None:
        staging_root = eval_root.parent / ".staging"
    bank_catalog_path = eval_root / "reward_bank.json"
    # Auto-construction is forecasting/generative only. TradingReward /
    # EventReward classes live in the curated trading and polymarket
    # families and must never be exposed to the Phase 3 reuse matcher
    # (LLM or exact-name) — otherwise a paper-side metric like
    # "BrierScore" can be reused against a trading class with the same
    # short alias and silently corrupt the downstream evaluator bundle.
    bank_catalog_raw = list(load_reward_catalog(bank_catalog_path))
    bank_catalog = [
        dict(entry, source_kind="curated")
        for entry in bank_catalog_raw
        if str(entry.get("base_class", "Loss")).strip() == "Loss"
    ]
    bank_catalog_excluded = len(bank_catalog_raw) - len(bank_catalog)
    logger.info(
        "%s load reward bank catalog (read-only) entries=%d excluded_non_loss=%d from=%s",
        PHASE3_TAG,
        len(bank_catalog),
        bank_catalog_excluded,
        bank_catalog_path,
    )

    if generated_dir is None:
        generated_dir = eval_root / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    # As with datasets, staged reward slugs without catalog entries mark failed
    # attempts. Retry their LLM work only when explicitly requested.
    previously_attempted_reward_slugs: set[str] = set()
    staging_rewards_dir = staging_root / "rewards"
    if not overwrite and staging_rewards_dir.exists():
        previously_attempted_reward_slugs = {
            entry.name for entry in staging_rewards_dir.iterdir() if entry.is_dir()
        }
    skip_previously_failed = (
        not overwrite and not retry_failed_attempts
    )
    if previously_attempted_reward_slugs:
        logger.info(
            "%s found %d previously-attempted reward slug(s) in staging=%s; "
            "their LLM fuzzy reuse check will be skipped on this run "
            "(skip_previously_failed=%s; pass --retry-failed-attempts or "
            "--overwrite to retry)",
            PHASE3_TAG,
            len(previously_attempted_reward_slugs),
            staging_rewards_dir,
            skip_previously_failed,
        )
    generated_catalog_path = generated_dir / "generated_rewards.json"
    # Same Loss-only filter as the bank catalog — defence in depth. The
    # Phase 3 codegen template emits Loss subclasses only, but a stale
    # or corrupted generated_rewards.json must not be load-bearing.
    generated_catalog_raw = list(load_reward_catalog(generated_catalog_path))
    generated_catalog = [
        dict(entry, source_kind="generated")
        for entry in generated_catalog_raw
        if str(entry.get("base_class", "Loss")).strip() == "Loss"
    ]
    generated_catalog_excluded = (
        len(generated_catalog_raw) - len(generated_catalog)
    )
    logger.info(
        "%s load generated reward catalog entries=%d excluded_non_loss=%d from=%s",
        PHASE3_TAG,
        len(generated_catalog),
        generated_catalog_excluded,
        generated_catalog_path,
    )

    combined_catalog = bank_catalog + generated_catalog

    effective_cap = written_cap
    if written_cap is not None:
        existing_generated_total = len(generated_catalog)
        existing_generated = sum(
            1 for entry in generated_catalog if scope_match(entry, scope_ids)
        )
        effective_cap = max(0, written_cap - existing_generated)
        logger.info(
            "%s absolute cap: total_cap=%d existing_generated_in_scope=%d "
            "(generated_total=%d) bank=%d scope_filter=%s remaining_budget=%d",
            PHASE3_TAG,
            written_cap,
            existing_generated,
            existing_generated_total,
            len(bank_catalog),
            scope_ids or "ALL",
            effective_cap,
        )
        if effective_cap <= 0:
            logger.info(
                "%s cap already met for new reward writes total_cap=%d existing_generated_in_scope=%d; continuing in reuse-only mode",
                PHASE3_TAG,
                written_cap,
                existing_generated,
            )

    aggregation = aggregate_rewards(
        task_summaries,
        existing_catalog=combined_catalog,
        llm=llm,
        debug_dir=summary_dir,
    )
    rewards = aggregation.candidates
    if overwrite:
        _reset_paper_reward_runtime_fields(
            research_root=research_root,
            task_dirs=_task_dirs_from_summaries(task_summaries),
        )
    logger.info(
        "%s aggregate candidate rewards raw_mentions=%d unique_named_candidates=%d from_paper_summaries=%d into=%s "
        "(reward mentions are clustered by exact name; unique_named_candidates is the count of distinct reward names)",
        PHASE3_TAG,
        aggregation.raw_mentions_count,
        aggregation.candidate_count,
        len(task_summaries),
        eval_root,
    )
    written: list[object] = []
    reused_count = 0
    skipped_written_cap_count = 0
    skipped_previously_failed_count = 0
    not_approved_count = 0
    log_stage(
        logger,
        "%s processing %d reward candidates",
        PHASE3_TAG,
        len(rewards),
    )
    for reward_index, reward in enumerate(rewards, start=1):
        log_stage(
            logger,
            "%s [%d/%d] candidate=%s papers=%s",
            PHASE3_TAG,
            reward_index,
            len(rewards),
            reward.name,
            format_paper_refs(reward.task_dirs) or "(none)",
        )
        log_detail(
            logger,
            "%s candidate detail name=%s mention_count=%s supporting_scopes=%s",
            PHASE3_TAG,
            reward.name,
            reward.frequency,
            ",".join(reward.supporting_scope_ids),
        )
        # At the cap or after a failed attempt, allow exact-name reuse without
        # spending another fuzzy-match call.
        reuse_only_mode = effective_cap is not None and len(written) >= effective_cap
        reward_slug_for_check = slugify(reward.name)
        already_attempted = (
            reward_slug_for_check in previously_attempted_reward_slugs
        )
        skip_llm_fuzzy_match = reuse_only_mode or already_attempted
        if already_attempted and not reuse_only_mode:
            log_detail(
                logger,
                "%s   -> skip_llm_fuzzy_match candidate=%s reason=previously_attempted "
                "(staging dir at %s)",
                PHASE3_TAG,
                reward.name,
                staging_rewards_dir / reward_slug_for_check,
            )
        reused, matched_entry, reason = can_reuse_existing_reward(
            reward, combined_catalog, None if skip_llm_fuzzy_match else llm
        )
        # TradingReward / EventReward classes are filtered out of
        # ``bank_catalog`` and ``generated_catalog`` at load time, so
        # ``can_reuse_existing_reward`` cannot match against them and no
        # downstream guard is needed here.
        if reused:
            matched_name = str(matched_entry.get("name", "")).strip() if matched_entry else ""
            matched_source_kind = str(matched_entry.get("source_kind", "")).strip() if matched_entry else ""
            # Tag generated reward entries with this candidate's scopes for
            # cap accounting. Curated/bank entries are scope-universal and
            # need no tagging.
            if matched_source_kind == "generated" and matched_name:
                _merge_reward_scope_ids_into_catalog(
                    generated_catalog=generated_catalog,
                    generated_catalog_path=generated_catalog_path,
                    reward_name=matched_name,
                    scope_ids_to_add=_candidate_scope_ids(reward),
                )
            reward_script_path = (
                _resolve_reused_reward_script_path(
                    eval_root=eval_root,
                    generated_dir=generated_dir,
                    matched_entry=matched_entry,
                )
                if matched_entry is not None
                else ""
            )
            generated_successful = bool(reward_script_path)
            spec_for_entry = _spec_from_catalog_entry(matched_entry)
            _update_paper_reward_entries(
                research_root=research_root,
                task_dirs=reward.task_dirs,
                reward_name=reward.name,
                reused=True,
                generated_successful=generated_successful,
                reward_script_path=reward_script_path,
                spec=spec_for_entry,
            )
            reused_count += 1
            log_stage(
                logger,
                "%s   -> reused (matched=%s source=%s reason=%s reward_script_path=%s)",
                PHASE3_TAG,
                matched_name,
                matched_source_kind,
                truncate_oneline(reason),
                reward_script_path or "<missing>",
            )
            continue

        # Reuse did not match. If a staging dir already exists for this
        # reward slug, the previous attempt did not pass review (no entry
        # in the generated catalog). Skip by default to avoid burning LLM
        # credits regenerating the same code; --retry-failed-attempts or
        # --overwrite opts back in.
        if (
            skip_previously_failed
            and reward_slug_for_check in previously_attempted_reward_slugs
        ):
            skipped_previously_failed_count += 1
            staging_path = staging_rewards_dir / reward_slug_for_check
            log_stage(
                logger,
                "%s   -> skipped (previously_failed staging=%s; pass "
                "--retry-failed-attempts to retry)",
                PHASE3_TAG,
                staging_path,
            )
            _update_paper_reward_entries(
                research_root=research_root,
                task_dirs=reward.task_dirs,
                reward_name=reward.name,
                reused=False,
                generated_successful=False,
                reward_script_path="",
            )
            continue

        if effective_cap is not None and len(written) >= effective_cap:
            skipped_written_cap_count += 1
            log_stage(
                logger,
                "%s   -> skipped (written_cap_reached total_cap=%s effective_cap=%d)",
                PHASE3_TAG,
                written_cap,
                effective_cap,
            )
            _update_paper_reward_entries(
                research_root=research_root,
                task_dirs=reward.task_dirs,
                reward_name=reward.name,
                reused=False,
                generated_successful=False,
                reward_script_path="",
            )
            continue

        result = write_reward_asset(
            eval_root,
            reward,
            llm=llm,
            staging_root=staging_root,
            reward_generation_rounds=reward_generation_rounds,
            generated_dir=generated_dir,
        )
        approved = bool(result.get("approved", False))
        catalog_updated = False
        reward_script_path = str(result.get("script_path", "")).strip()
        if approved:
            written.append(reward)
            candidate_scopes = _candidate_scope_ids(reward)
            catalog_updated = upsert_reward_catalog_entry(
                generated_catalog,
                reward,
                spec=result.get("spec"),
                scope_ids=candidate_scopes,
            )
            if catalog_updated:
                # Strip the transient source_kind annotation injected at load
                # time so it doesn't leak into the on-disk catalog file.
                on_disk = [
                    {k: v for k, v in entry.items() if k != "source_kind"}
                    for entry in generated_catalog
                ]
                generated_catalog_path.write_text(
                    json.dumps(on_disk, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.info(
                    "%s update generated reward catalog path=%s total_entries_now=%d",
                    PHASE3_TAG,
                    generated_catalog_path,
                    len(generated_catalog),
                )
            combined_catalog.append(
                {
                    "name": reward.name,
                    "description": reward.description,
                    "source_kind": "generated",
                }
            )
        else:
            not_approved_count += 1
            reward_script_path = ""
        spec_for_entry = _spec_from_write_result(result) if approved else None
        _update_paper_reward_entries(
            research_root=research_root,
            task_dirs=reward.task_dirs,
            reward_name=reward.name,
            reused=False,
            generated_successful=approved,
            reward_script_path=reward_script_path,
            spec=spec_for_entry,
        )
        log_stage(
            logger,
            "%s   -> %s (catalog_updated=%s runtime_reward_path=%s)",
            PHASE3_TAG,
            "materialized" if approved else "not_approved",
            catalog_updated,
            reward_script_path,
        )

    return {
        "written": written,
        "raw_mentions_count": aggregation.raw_mentions_count,
        "candidate_count": aggregation.candidate_count,
        "reused": reused_count,
        "skipped_written_cap": skipped_written_cap_count,
        "skipped_previously_failed": skipped_previously_failed_count,
    }


def _resolve_dataset_generation_rounds(cfg: AppConfig, override: int | None) -> int:
    rounds = int(
        override if override is not None else cfg.benchmark.dataset_generation_rounds
    )
    if rounds < 1:
        raise ValueError("dataset_generation_rounds must be >= 1")
    return rounds


def _resolve_reward_generation_rounds(cfg: AppConfig, override: int | None) -> int:
    rounds = int(
        override if override is not None else cfg.benchmark.reward_generation_rounds
    )
    if rounds < 1:
        raise ValueError("reward_generation_rounds must be >= 1")
    return rounds


def _filter_task_summaries(
    task_summaries: list[tuple[str, dict[str, object]]],
    scope_ids: list[str] | None,
) -> list[tuple[str, dict[str, object]]]:
    if not scope_ids:
        return task_summaries
    targets = {value.strip() for value in scope_ids if value and value.strip()}
    if not targets:
        return task_summaries
    filtered: list[tuple[str, dict[str, object]]] = []
    for task_dir, summary in task_summaries:
        scope_id = str(summary.get("_scope_id", "")).strip()
        if scope_id in targets:
            filtered.append((task_dir, summary))
    return filtered


def _drop_trading_papers(
    task_summaries: list[tuple[str, dict[str, object]]],
) -> tuple[list[tuple[str, dict[str, object]]], int]:
    """Drop papers whose summary.task_family routes through Phase 1c triage.

    Trading-flavored papers (`trading`, `realtime_trading`, `realtime_forecasting`)
    are handled by `triage_trading_papers` (Phase 1c) and bound to curated
    overlays under `tasks/routed/`. Phase 2 dataset codegen and Phase 3 reward
    harvesting have nothing to do for them — the curated tasks bring their own
    data providers and reward bank.
    """
    from openfinai_pipeline.papers.triage import is_trading_family

    kept: list[tuple[str, dict[str, object]]] = []
    dropped = 0
    for task_dir, summary in task_summaries:
        family = summary.get("task_family") if isinstance(summary, dict) else None
        if is_trading_family(family if isinstance(family, str) else None):
            dropped += 1
            continue
        kept.append((task_dir, summary))
    return kept, dropped
