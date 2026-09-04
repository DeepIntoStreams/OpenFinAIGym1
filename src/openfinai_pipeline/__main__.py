import argparse
import sys
from typing import Sequence

from dotenv import load_dotenv

from openfinai_pipeline.benchmark.workflow import construct_tasks
from openfinai_pipeline.corpus.workflow import (
    construct_dataset,
    construct_rewards,
)
from openfinai_pipeline.papers.pipeline import (
    DEFAULT_OVERLAY_ROOT,
    import_manual_papers,
    run_scraping,
    summarize_papers,
    triage_trading_papers,
)
from openfinai_pipeline.papers.triage import DEFAULT_PDF_TEXT_MAX_CHARS
from openfinai_pipeline.summary.prompt_context import DEFAULT_SUMMARY_EXCERPT_CHARS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m openfinai_pipeline",
        description="OpenFinGym task-construction pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape = subparsers.add_parser(
        "scrape",
        help="scrape papers and persist accepted paper.json/source.pdf artifacts",
    )
    _add_common_scrape_options(scrape)
    scrape.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help=(
            "Re-judge papers already seen in this scope's index.json. "
            "By default, papers whose paper_id appears in either entries "
            "(previously accepted) or seen_rejected (previously rejected) "
            "are filtered out post-collection so prefilter/sift LLM calls "
            "and PDF downloads are not re-spent on them. With --overwrite, "
            "the dedupe filter is bypassed and accepted re-judgments reuse "
            "the existing paperN/ slot (overwriting source.pdf in place — "
            "no duplicate directories are created)."
        ),
    )
    _add_common_llm_options(scrape)

    summarize = subparsers.add_parser(
        "summarize-paper",
        help=(
            "summarize accepted papers into paper.json summary payloads "
            "using local source.pdf files"
        ),
    )
    summarize.add_argument("--config", dest="config_path", default=None)
    summarize.add_argument("--scopes", nargs="*", default=None)
    summarize.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
    )
    summarize.add_argument(
        "--summary-excerpt-chars",
        dest="summary_excerpt_chars",
        type=int,
        default=DEFAULT_SUMMARY_EXCERPT_CHARS,
    )
    _add_common_llm_options(summarize)

    triage = subparsers.add_parser(
        "triage-trading-papers",
        help=(
            "Phase 1c: route trading-classified papers to curated tasks via "
            "per-paper LLM triage; writes trading_triage.json sidecars and "
            "tasks/routed/<id>/ overlays"
        ),
    )
    triage.add_argument("--config", dest="config_path", default=None)
    triage.add_argument("--scopes", nargs="*", default=None)
    triage.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Re-triage papers that already have a trading_triage.json sidecar.",
    )
    triage.add_argument(
        "--pdf-text-max-chars",
        dest="pdf_text_max_chars",
        type=int,
        default=DEFAULT_PDF_TEXT_MAX_CHARS,
        help=(
            "Truncate the PDF text passed to the triage LLM to this many "
            f"characters (default: {DEFAULT_PDF_TEXT_MAX_CHARS:,}). Set to 0 "
            "to disable truncation (may exceed model context window for very "
            "long papers)."
        ),
    )
    triage.add_argument(
        "--overlay-root",
        dest="overlay_root",
        default=None,
        help=(
            f"Directory under which to write per-paper routed overlays "
            f"(default: {DEFAULT_OVERLAY_ROOT}). Each paper gets a thin "
            "subdirectory containing task.toml + instruction.md + __init__.py "
            "+ triage_record.json."
        ),
    )
    triage.add_argument(
        "--routes-written-cap",
        dest="routes_written_cap",
        type=int,
        default=None,
        help=(
            "Absolute total routed/partial_match papers across the corpus, "
            "scope-aware against existing trading_triage.json sidecars in "
            "each requested scope. Once the remaining budget is exhausted, "
            "subsequent trading papers are skipped without firing the "
            "triage LLM (counted as skipped_written_cap). no_match and "
            "novel_required outcomes write no artifact and do not consume "
            "the budget. Re-running with the same cap adds zero."
        ),
    )
    _add_common_llm_options(triage)

    import_manual = subparsers.add_parser(
        "import-manual-papers",
        help="import local PDFs into paper.json/source.pdf artifacts under paperN directories",
    )
    import_manual.add_argument("--config", dest="config_path", default=None)
    import_manual.add_argument(
        "--mapping",
        action="append",
        default=[],
        help="Repeated <scope_id>=<folder> mapping",
    )
    import_manual.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
    )

    construct_dataset_cmd = subparsers.add_parser(
        "construct-dataset",
        help="construct benchmark dataset assets from paper summaries",
    )
    construct_dataset_cmd.add_argument("--config", dest="config_path", default=None)
    construct_dataset_cmd.add_argument(
        "--datasets-written-cap",
        type=int,
        default=None,
        help="Max total datasets including previously written",
    )
    construct_dataset_cmd.add_argument("--scopes", nargs="*", default=None)
    construct_dataset_cmd.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
    )
    _add_common_llm_options(construct_dataset_cmd)
    _add_dataset_generation_rounds_option(construct_dataset_cmd)
    _add_retry_failed_attempts_option(construct_dataset_cmd)

    construct_rewards_cmd = subparsers.add_parser(
        "construct-rewards",
        help="construct benchmark reward/eval assets from paper summaries",
    )
    construct_rewards_cmd.add_argument("--config", dest="config_path", default=None)
    construct_rewards_cmd.add_argument(
        "--rewards-written-cap",
        type=int,
        default=None,
        help="Max total rewards including previously written",
    )
    construct_rewards_cmd.add_argument("--scopes", nargs="*", default=None)
    construct_rewards_cmd.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
    )
    _add_common_llm_options(construct_rewards_cmd)
    _add_reward_generation_rounds_option(construct_rewards_cmd)
    _add_retry_failed_attempts_option(construct_rewards_cmd)

    construct_tasks_cmd = subparsers.add_parser(
        "construct-benchmark-tasks",
        help="generate canonical benchmark task packages from harvested papers",
    )
    construct_tasks_cmd.add_argument("--config", dest="config_path", default=None)
    construct_tasks_cmd.add_argument(
        "--tasks-written-cap",
        dest="tasks_written_cap",
        type=int,
        default=None,
        help="Max total installed tasks including existing",
    )
    construct_tasks_cmd.add_argument("--scopes", nargs="*", default=None)
    construct_tasks_cmd.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
    )
    _add_common_llm_options(construct_tasks_cmd)
    _add_task_generation_rounds_option(construct_tasks_cmd)
    _add_loader_generation_rounds_option(construct_tasks_cmd)
    _add_retry_failed_attempts_option(construct_tasks_cmd)

    construct_tasks_alias = subparsers.add_parser(
        "construct-tasks",
        help="alias for construct-benchmark-tasks",
    )
    construct_tasks_alias.add_argument("--config", dest="config_path", default=None)
    construct_tasks_alias.add_argument(
        "--tasks-written-cap",
        dest="tasks_written_cap",
        type=int,
        default=None,
        help="Max total installed tasks including existing",
    )
    construct_tasks_alias.add_argument("--scopes", nargs="*", default=None)
    construct_tasks_alias.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
    )
    _add_common_llm_options(construct_tasks_alias)
    _add_task_generation_rounds_option(construct_tasks_alias)
    _add_loader_generation_rounds_option(construct_tasks_alias)
    _add_retry_failed_attempts_option(construct_tasks_alias)

    run = subparsers.add_parser(
        "run-all",
        help="run scrape -> summarize-paper -> construct-dataset -> construct-rewards -> construct-tasks",
    )
    _add_common_scrape_options(run)
    run.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
    )
    run.add_argument(
        "--datasets-written-cap",
        type=int,
        default=None,
        help="Max total datasets including previously written",
    )
    run.add_argument(
        "--rewards-written-cap",
        type=int,
        default=None,
        help="Max total rewards including previously written",
    )
    run.add_argument(
        "--tasks-written-cap",
        dest="tasks_written_cap",
        type=int,
        default=None,
        help="Max total installed tasks including existing",
    )
    run.add_argument(
        "--summary-excerpt-chars",
        dest="summary_excerpt_chars",
        type=int,
        default=DEFAULT_SUMMARY_EXCERPT_CHARS,
    )
    run.add_argument(
        "--pdf-text-max-chars",
        dest="pdf_text_max_chars",
        type=int,
        default=DEFAULT_PDF_TEXT_MAX_CHARS,
        help=(
            "Phase 1c (triage): truncate the PDF text passed to the triage "
            f"LLM to this many characters (default: {DEFAULT_PDF_TEXT_MAX_CHARS:,})."
        ),
    )
    run.add_argument(
        "--overlay-root",
        dest="overlay_root",
        default=None,
        help=(
            f"Phase 1c (triage): directory under which routed overlays go "
            f"(default: {DEFAULT_OVERLAY_ROOT})."
        ),
    )
    run.add_argument(
        "--routes-written-cap",
        dest="routes_written_cap",
        type=int,
        default=None,
        help=(
            "Phase 1c: absolute total routed/partial_match papers across "
            "the corpus including previously-routed; scope-aware against "
            "existing triage sidecars."
        ),
    )
    _add_common_llm_options(run)
    _add_dataset_generation_rounds_option(run)
    _add_reward_generation_rounds_option(run)
    _add_task_generation_rounds_option(run)
    _add_loader_generation_rounds_option(run)
    _add_retry_failed_attempts_option(run)

    return parser


def _add_common_scrape_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", dest="config_path", default="./config/base.yaml")
    parser.add_argument("--scopes", nargs="*", default=None)
    parser.add_argument("--since", default=None, help="YYYY-MM-DD")
    parser.add_argument("--until", default=None, help="YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-accept",
        dest="max_accepts",
        type=int,
        default=None,
        help="Max total accepted papers per scope including existing",
    )


def _add_common_llm_options(parser: argparse.ArgumentParser) -> None:
    """LLM CLI overrides.

    All LLM arg defaults are ``None`` so the YAML (``config/base.yaml``'s
    ``llm:`` block, including any ``llm.per_phase.*`` overrides) wins
    when the operator omits them. Pass an explicit value to force-override
    the YAML for that phase. Precedence: CLI flag > per_phase > global llm.*.
    """
    parser.add_argument(
        "--provider",
        default=None,
        help="Override llm.provider for this run; default is base.yaml llm.provider.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model for the resolved provider. If unset, uses the per-phase override (if any) or the provider's default from llm.providers.",
    )
    parser.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=None,
        help="Override llm.max_tokens for this run.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override llm.temperature for this run.",
    )
    parser.add_argument(
        "--llm-max-retries",
        dest="llm_max_retries",
        type=int,
        default=None,
        help="Max retries on a single LLM provider call against transient errors (429, 5xx, network timeouts). Overrides llm.retry.max_retries.",
    )
    parser.add_argument(
        "--timeout-sec",
        dest="timeout_sec",
        type=float,
        default=None,
        help="Per-call provider timeout. Overrides llm.providers[provider].timeout_sec.",
    )
    parser.add_argument(
        "--tool-max-turns",
        dest="tool_max_turns",
        type=int,
        default=None,
        help="Override llm.tool_max_turns for this run.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help=(
            "show per-item interior INFO on the terminal in addition to "
            "stage markers. Default tier shows only stage markers + "
            "warnings/errors. Full forensic detail (incl. LLM response "
            "bodies and per-paper detail dumps) is always written to the "
            "per-run log file under data/pipeline_output/logs/, never to "
            "the terminal."
        ),
    )


def _add_dataset_generation_rounds_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-generation-rounds",
        dest="dataset_generation_rounds",
        type=int,
        default=None,
        help=(
            "Override the number of codegen/fix rounds for Phase 2 dataset "
            "download script generation (default from "
            "config.benchmark.dataset_generation_rounds, currently 2)"
        ),
    )


def _add_reward_generation_rounds_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reward-generation-rounds",
        dest="reward_generation_rounds",
        type=int,
        default=None,
        help=(
            "Override the number of codegen/fix rounds for Phase 3 "
            "reward_fn generation (default from "
            "config.benchmark.reward_generation_rounds, currently 3)"
        ),
    )


def _add_task_generation_rounds_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task-generation-rounds",
        dest="task_generation_rounds",
        type=int,
        default=None,
        help=(
            "Override the number of codegen/fix rounds for Phase 4 task "
            "assembly retries (default from "
            "config.benchmark.task_generation_rounds, currently 3)"
        ),
    )


def _add_loader_generation_rounds_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--loader-generation-rounds",
        dest="loader_generation_rounds",
        type=int,
        default=None,
        help=(
            "Override the number of per-dataset load.py codegen/fix rounds "
            "(independent from --generation-rounds; default from "
            "config.benchmark.loader_generation_rounds, currently 3)"
        ),
    )


def _add_retry_failed_attempts_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retry-failed-attempts",
        dest="retry_failed_attempts",
        action="store_true",
        help=(
            "Retry candidates whose previous run left a staging directory "
            "behind without a corresponding success record (catalog entry "
            "for datasets/rewards, installed task for Phase 4). By default "
            "such previously-failed candidates are skipped on rerun to "
            "avoid burning LLM credits and network on known failures. "
            "--overwrite also forces retry."
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    if args.command == "scrape":
        summary = run_scraping(
            config_path=args.config_path,
            scopes_override=args.scopes,
            since=args.since,
            until=args.until,
            limit=args.limit,
            max_accepts=args.max_accepts,
            log_practice="scrape",
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            llm_max_retries=args.llm_max_retries,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
            verbose=args.verbose,
            overwrite=args.overwrite,
        )
        print(
            (
                f"run_id={summary.run_id} status={summary.status} "
                f"accepted={summary.total_accepted} "
                f"persisted={summary.total_tasks_constructed}"
            )
        )
        return 0

    if args.command == "summarize-paper":
        result = summarize_papers(
            config_path=args.config_path,
            scopes_override=args.scopes,
            overwrite=args.overwrite,
            summary_excerpt_chars=args.summary_excerpt_chars,
            log_practice="summarize_paper",
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            llm_max_retries=args.llm_max_retries,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
            verbose=args.verbose,
        )
        print(
            (
                f"run_id={result.get('run_id', '')} "
                f"status={result.get('status', '')} "
                f"summarized={result.get('summarized', 0)} "
                f"skipped_existing={result.get('skipped_existing', 0)} "
                f"failed={result.get('failed', 0)}"
            )
        )
        return 0

    if args.command == "triage-trading-papers":
        result = triage_trading_papers(
            config_path=args.config_path,
            scopes_override=args.scopes,
            overwrite=args.overwrite,
            pdf_text_max_chars=args.pdf_text_max_chars,
            overlay_root=args.overlay_root,
            routes_written_cap=args.routes_written_cap,
            log_practice="triage_trading_papers",
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            llm_max_retries=args.llm_max_retries,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
            verbose=args.verbose,
        )
        print(
            (
                f"run_id={result.get('run_id', '')} "
                f"status={result.get('status', '')} "
                f"triaged={result.get('triaged', 0)} "
                f"routed={result.get('routed', 0)} "
                f"partial={result.get('partial_match', 0)} "
                f"no_match={result.get('no_match', 0)} "
                f"novel={result.get('novel_required', 0)} "
                f"skipped_existing={result.get('skipped_existing', 0)} "
                f"skipped_cap={result.get('skipped_written_cap', 0)} "
                f"failed={result.get('failed', 0)}"
            )
        )
        return 0

    if args.command == "import-manual-papers":
        result = import_manual_papers(
            config_path=args.config_path,
            mappings=args.mapping,
            overwrite=args.overwrite,
        )
        print(
            (
                f"run_id={result.get('run_id', '')} "
                f"status={result.get('status', '')} "
                f"imported={result.get('imported', 0)} "
                f"skipped_existing={result.get('skipped_existing', 0)} "
                f"overwritten={result.get('overwritten', 0)} "
                f"failed={result.get('failed', 0)}"
            )
        )
        return 0

    if args.command == "construct-dataset":
        result = construct_dataset(
            args.config_path,
            datasets_written_cap=args.datasets_written_cap,
            scope_ids=args.scopes,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            llm_max_retries=args.llm_max_retries,
            timeout_sec=args.timeout_sec,
            tool_max_turns=args.tool_max_turns,
            dataset_generation_rounds=args.dataset_generation_rounds,
            overwrite=args.overwrite,
            retry_failed_attempts=args.retry_failed_attempts,
            verbose=args.verbose,
        )
        print(
            (
                f"datasets={result.get('datasets_written', 0)} "
                f"reused={result.get('datasets_reused', 0)} "
                f"raw_mentions={result.get('raw_mentions', 0)} "
                f"candidates={result.get('candidates', 0)} "
                f"skipped_cap={result.get('datasets_skipped_written_cap', 0)} "
                f"skipped_previously_failed={result.get('datasets_skipped_previously_failed', 0)}"
            )
        )
        return 0

    if args.command == "construct-rewards":
        result = construct_rewards(
            args.config_path,
            rewards_written_cap=args.rewards_written_cap,
            scope_ids=args.scopes,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            llm_max_retries=args.llm_max_retries,
            timeout_sec=args.timeout_sec,
            tool_max_turns=args.tool_max_turns,
            reward_generation_rounds=args.reward_generation_rounds,
            overwrite=args.overwrite,
            retry_failed_attempts=args.retry_failed_attempts,
            verbose=args.verbose,
        )
        print(
            (
                f"rewards={result.get('rewards_written', 0)} "
                f"reused={result.get('rewards_reused', 0)} "
                f"raw_mentions={result.get('raw_mentions', 0)} "
                f"unique_named_candidates={result.get('unique_named_candidates', 0)} "
                f"skipped_cap={result.get('rewards_skipped_written_cap', 0)} "
                f"skipped_previously_failed={result.get('rewards_skipped_previously_failed', 0)}"
            )
        )
        return 0

    if args.command in {"construct-benchmark-tasks", "construct-tasks"}:
        construct_tasks(
            args.config_path,
            tasks_written_cap=args.tasks_written_cap,
            scope_ids=args.scopes,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            llm_max_retries=args.llm_max_retries,
            tool_max_turns=args.tool_max_turns,
            task_generation_rounds=args.task_generation_rounds,
            loader_generation_rounds=args.loader_generation_rounds,
            timeout_sec=args.timeout_sec,
            overwrite=args.overwrite,
            retry_failed_attempts=args.retry_failed_attempts,
            verbose=args.verbose,
        )
        return 0

    if args.command == "run-all":
        summary = run_scraping(
            config_path=args.config_path,
            scopes_override=args.scopes,
            since=args.since,
            until=args.until,
            limit=args.limit,
            max_accepts=args.max_accepts,
            log_practice="run-all_scrape",
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            llm_max_retries=args.llm_max_retries,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
            verbose=args.verbose,
            overwrite=args.overwrite,
        )
        summarize_result = summarize_papers(
            config_path=args.config_path,
            scopes_override=args.scopes,
            overwrite=args.overwrite,
            summary_excerpt_chars=args.summary_excerpt_chars,
            log_practice="run-all_summarize_paper",
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            llm_max_retries=args.llm_max_retries,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
            verbose=args.verbose,
        )
        triage_result = triage_trading_papers(
            config_path=args.config_path,
            scopes_override=args.scopes,
            overwrite=args.overwrite,
            pdf_text_max_chars=args.pdf_text_max_chars,
            overlay_root=args.overlay_root,
            routes_written_cap=args.routes_written_cap,
            log_practice="run-all_triage_trading_papers",
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            llm_max_retries=args.llm_max_retries,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
            verbose=args.verbose,
        )
        ds = construct_dataset(
            args.config_path,
            datasets_written_cap=args.datasets_written_cap,
            scope_ids=args.scopes,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            llm_max_retries=args.llm_max_retries,
            timeout_sec=args.timeout_sec,
            tool_max_turns=args.tool_max_turns,
            dataset_generation_rounds=args.dataset_generation_rounds,
            overwrite=args.overwrite,
            retry_failed_attempts=args.retry_failed_attempts,
            verbose=args.verbose,
        )
        rw = construct_rewards(
            args.config_path,
            rewards_written_cap=args.rewards_written_cap,
            scope_ids=args.scopes,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            llm_max_retries=args.llm_max_retries,
            timeout_sec=args.timeout_sec,
            tool_max_turns=args.tool_max_turns,
            reward_generation_rounds=args.reward_generation_rounds,
            overwrite=args.overwrite,
            retry_failed_attempts=args.retry_failed_attempts,
            verbose=args.verbose,
        )
        tasks = construct_tasks(
            args.config_path,
            tasks_written_cap=args.tasks_written_cap,
            scope_ids=args.scopes,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            llm_max_retries=args.llm_max_retries,
            tool_max_turns=args.tool_max_turns,
            task_generation_rounds=args.task_generation_rounds,
            loader_generation_rounds=args.loader_generation_rounds,
            timeout_sec=args.timeout_sec,
            overwrite=args.overwrite,
            retry_failed_attempts=args.retry_failed_attempts,
            verbose=args.verbose,
        )
        print(
            (
                f"run_id={summary.run_id} scrape_status={summary.status} "
                f"summarized={summarize_result.get('summarized', 0)} "
                f"triage_routed={triage_result.get('routed', 0)} "
                f"triage_partial={triage_result.get('partial_match', 0)} "
                f"triage_no_match={triage_result.get('no_match', 0)} "
                f"datasets={ds.get('datasets_written', 0)} "
                f"rewards={rw.get('rewards_written', 0)} "
                f"tasks={tasks.get('tasks_implemented', 0)} "
                f"agent_smoke_failed={tasks.get('tasks_failed_agent_smoke', 0)} "
                f"smoke_failed={tasks.get('tasks_smoke_validation_failed', 0)}"
            )
        )
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
