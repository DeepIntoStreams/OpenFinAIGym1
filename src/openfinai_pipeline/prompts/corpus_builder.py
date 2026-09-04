import json
import logging
import threading
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from openfinai_pipeline.corpus.manifest import (
    FORMAT_VOCAB,
    ROLE_VOCAB,
    dataset_label_schema,
)
from openfinai_pipeline.utils.logging import log_detail

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "templates"

# Per-field character cap for variable prompt fields.
_MAX_FIELD_CHARS = 200_000

# Dedup cache: keep the user-visible warning for the first (field_name, len,
# max_chars) tuple per process, demote duplicates to file-only log_detail.
# Same paper context flows into both codegen and review prompts per round.
_TRUNCATE_WARN_LOCK = threading.Lock()
_TRUNCATE_WARN_SEEN: set[tuple[str, int, int]] = set()


def _truncate(
    text: str,
    max_chars: int = _MAX_FIELD_CHARS,
    *,
    keep_tail: bool = False,
    field_name: str = "",
) -> str:
    """Truncate *text* to *max_chars*.

    When *keep_tail* is True the tail is preserved (useful for execution logs
    where errors appear at the end).  Otherwise the head is kept.
    """
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    key = (field_name or "(unknown)", len(text), max_chars)
    with _TRUNCATE_WARN_LOCK:
        first_occurrence = key not in _TRUNCATE_WARN_SEEN
        if first_occurrence:
            _TRUNCATE_WARN_SEEN.add(key)
    if first_occurrence:
        logger.warning(
            "Prompt field %s truncated from %s to %s chars (dropped %s chars)",
            field_name or "(unknown)",
            f"{len(text):,}",
            f"{max_chars:,}",
            f"{dropped:,}",
        )
    else:
        log_detail(
            logger,
            "Prompt field %s truncated from %s to %s chars (dropped %s chars) [duplicate]",
            field_name or "(unknown)",
            f"{len(text):,}",
            f"{max_chars:,}",
            f"{dropped:,}",
        )
    if keep_tail:
        head_chars = max_chars // 5
        tail_chars = max_chars - head_chars - 80  # room for marker
        marker = (
            f"\n\n... [{len(text) - head_chars - tail_chars:,} chars truncated] ...\n\n"
        )
        return text[:head_chars] + marker + text[-tail_chars:]
    return text[:max_chars] + f"\n... [{dropped:,} chars truncated]"


def build_dataset_reuse_prompt(
    *,
    candidate_name: str,
    candidate_aliases: list[str],
    candidate_description: str,
    existing_datasets: list[dict[str, str]],
    candidate_reproducibility_context: str = "",
) -> str:
    catalog_json = _truncate(
        json.dumps(existing_datasets, ensure_ascii=False),
        field_name="existing_datasets",
    )
    aliases = ", ".join(a for a in candidate_aliases if a) or "(none)"
    desc_block = (
        _truncate(
            (candidate_description or "").strip(), field_name="candidate_description"
        )
        or "(none)"
    )
    # 4 KB cap is ~2x the longest experiments block; per-candidate prompt only
    # needs the experiments/target/label/split summary, not full paper context.
    repro_block = (
        _truncate(
            (candidate_reproducibility_context or "").strip(),
            max_chars=4_000,
            field_name="candidate_reproducibility_context",
        )
        or "(none)"
    )
    return f"""
You are deciding whether a newly extracted dataset can reuse an already existing local dataset.

Candidate dataset (the one currently being processed):
- name: {candidate_name}
- aliases: {aliases}
- description: {desc_block}
- candidate paper reproducibility context (experiments, evaluation target, label,
  split, features as the candidate paper uses the dataset):
{repro_block}

Existing local datasets. Each entry has two description fields with different semantics:
- `description` is paper-aspirational: how the original paper(s) describe the
  dataset family. Use it as context for what the entry is supposed to be.
- `downloaded_dataset_description` is the materialized payload on disk: what the
  acquisition script actually fetched, including any honest narrowing (subset of
  date range, sample of universe, computed-vs-vendor features, substituted
  auxiliary fields). Treat this as ground truth for what is currently available
  locally.
{catalog_json}

Decision policy:
1) reuse=true when the materialized payload contains data from which the
   candidate paper's evaluation target can be deterministically derived,
   AND the universe family, asset class, and granularity match the candidate's
   paper context.
2) The candidate paper does NOT need exact-match coverage of the existing
   entry, and the existing entry does NOT need to mention the candidate paper.
   Subset coverage of date range, a representative sample of the universe, and
   computed-from-payload feature substitutions are acceptable — these are the
   same narrowings a fresh acquisition round for the candidate would be
   approved with under the wall-clock budget. Honest narrowing recorded in
   `downloaded_dataset_description` is the standard, not full coverage.
3) Strong positive signals: exact dataset name, obvious rename, same source/
   provider/repository/landing page, same benchmark family, the materialized
   payload contains the column(s) needed to compute the candidate paper's
   target (or the target itself), and same instrument identifiers / asset class
   / frequency.
4) Reuse is NOT allowed when the universe family differs (e.g. crypto vs
   equities, US vs CN, futures vs spot), the asset class differs, the
   granularity differs (daily vs minute, tick vs bar), or the materialized
   payload lacks any column from which the candidate's target can be derived
   deterministically.
5) Reuse is NOT allowed when the existing entry is a distinct paper-specific
   derived benchmark, simulator, sandbox artifact, or hand-curated label set
   that is not interchangeable with the candidate's source.
6) Do not reuse based on topical similarity alone.
7) If uncertain, choose reuse=false.
8) matched_name must be one of existing dataset names when reuse=true.
9) In reason, name the candidate's evaluation target explicitly and explain
   how (or whether) it can be derived from the matched entry's
   `downloaded_dataset_description`, plus the universe / asset class /
   granularity match. Do not say only "semantically similar" or "same family".

Output JSON only:
- reason: string, detailed justification. Focus on target derivability against `downloaded_dataset_description`, plus universe / asset class / granularity match. Walk through each plausibly-matching existing entry; if none matches, say so explicitly.
- matched_name: string (empty string when reuse=false), the name of the existing dataset that can be reused when reuse=true.
- reuse: boolean. Must be consistent with `reason` and `matched_name`.
""".strip()


def dataset_reuse_schema() -> dict:
    return {
        "title": "DatasetReuseDecision",
        "description": "Determines whether a newly extracted dataset can reuse an existing dataset in the local catalog.",
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "matched_name": {"type": "string"},
            "reuse": {"type": "boolean"},
        },
        "required": ["reason", "matched_name", "reuse"],
        "additionalProperties": False,
    }


def _wall_clock_budget_block(timeout_sec: int | None) -> str:
    """Render the wall-clock-budget section, or empty string when no budget given.

    The same text is injected into both the codegen prompt and the reviewer
    prompt so both sides agree on what "narrowed scope" means and when it is
    acceptable. The exact second budget is interpolated rather than hard-coded
    so changing ``benchmark.download_timeout_sec`` in config flows through to
    the LLM without prompt edits.
    """
    if timeout_sec is None:
        return ""
    return f"""
Wall-clock budget
- The download script is killed after {int(timeout_sec)}s of wall-clock time.
- If acquiring the full paper-described coverage (full universe x full date range x full feature set) would not fit in this budget at modest network throughput, narrow the scope so the script can complete in time. Acceptable narrowings: a smaller universe (fewer instruments, or a representative sample), a shorter date range (a paper-relevant window or a recent slice), fewer feature columns, or a single representative split.
- Document any narrowing honestly in `downloaded_dataset_description` per the existing identity-narrowing guidance. The reviewer is told that narrowed-but-honest payloads are acceptable.
- Prefer batched or parallelized fetches when the provider supports them. Sequential per-instrument loops over hundreds or thousands of items typically do not fit in this budget; reshape the acquisition path before writing such a loop.
- A previous-round `timeout` failure category means the prior attempt blew the wall-clock budget. Treat that as a signal to materially narrow the scope for this round, not to retry the same approach with minor tweaks.
""".rstrip()


def _wall_clock_budget_review_bullet(
    timeout_sec: int | None, *, materialization_label: str
) -> str:
    """Render a review-criteria bullet that ties the budget to approve/reject.

    Returns an empty string when ``timeout_sec`` is None — in that mode the
    reviewer has no budget to reference, so emitting a "see the budget
    section" pointer would dangle. ``materialization_label`` is the noun the
    reviewer should narrow ("acquisition" for real, "materialization" for
    synthetic).
    """
    if timeout_sec is None:
        return ""
    return (
        "- WALL-CLOCK BUDGET: see the \"Wall-clock budget\" section above for the "
        "per-round wall-clock kill time. Approve a deliberately narrowed "
        f"{materialization_label} (smaller universe, shorter date range, batched "
        "fetches, single representative split) when the narrowing is honestly "
        "disclosed in `downloaded_dataset_description` per the three-tier "
        "task-completeness rule. Reject a round that retries the same wide scope "
        "after a prior `timeout` (failure category in the execution log) — the "
        "next round must materially narrow. Reject a script whose chosen scope "
        "obviously cannot complete within the budget at modest network "
        "throughput (e.g. sequential per-symbol loops over hundreds or thousands "
        "of symbols across many years), even when no prior `timeout` is on file "
        "yet."
    )


def _paper_links_block(paper_links: list[str] | None) -> str:
    links = [str(link).strip() for link in paper_links or [] if str(link).strip()]
    if not links:
        return "(none)"
    return "\n".join(f"- {link}" for link in links)


def build_write_download_code_prompt(
    dataset_name: str,
    dataset_description: str,
    dataset_kind: str,
    aliases: list[str],
    paper_links: list[str] | None = None,
    reproducibility_context: str = "",
    execution_log: str = "",
    previous_review: str = "",
    previous_script: str = "",
    previous_evidence: list[dict[str, object]] | None = None,
    failure_categories: list[str] | None = None,
    timeout_sec: int | None = None,
) -> str:
    dataset_kind_block = (dataset_kind or "").strip() or "real"
    aliases_str = ", ".join(aliases) if aliases else "(none)"
    desc_block = (
        _truncate((dataset_description or "").strip(), field_name="dataset_description")
        or "(none)"
    )
    reproducibility_block = (
        _truncate(
            (reproducibility_context or "").strip(),
            field_name="reproducibility_context",
        )
        or "(none)"
    )
    paper_links_block = _truncate(
        _paper_links_block(paper_links),
        field_name="paper_links",
    )
    log_block = (
        _truncate(execution_log.strip(), keep_tail=True, field_name="execution_log")
        or "(none)"
    )
    review_block = (
        _truncate(previous_review.strip(), field_name="previous_review") or "(none)"
    )
    prev_script_block = (
        _truncate(previous_script.strip(), field_name="previous_script") or "(none)"
    )
    evidence_block = _truncate(
        json.dumps(previous_evidence or [], ensure_ascii=False, indent=2),
        field_name="previous_evidence",
    ) or "[]"
    failure_block = ", ".join(failure_categories or []) or "(none)"
    budget_block = _wall_clock_budget_block(timeout_sec)
    return f"""
Task: write a Python script that acquires the target real dataset locally, or explicitly conclude that there is insufficient public evidence for a safe scriptable acquisition path.

Context
- Dataset name: {dataset_name}
- Dataset kind: {dataset_kind_block}
- Dataset description: {desc_block}
- Paper reproducibility context: {reproducibility_block}
- Paper direct links:
{paper_links_block}
- Aliases: {aliases_str}
- Previous round script:
{prev_script_block}

- Previous execution log:
{log_block}

- Previous round review:
{review_block}

- Accumulated evidence from all prior rounds:
{evidence_block}

- Observed failure categories from prior rounds:
{failure_block}
{budget_block}

Track
- This is the real-data acquisition track.
- The goal is to replicate the real dataset used in the paper's experiments as closely as the evidence supports.
- Do not replace missing real data with synthetic, mocked, sampled, or placeholder stand-ins.

Paper-provided links and implementation pointers
- Treat any GitHub link, dataset landing page, download URL, repository link, appendix code pointer, notebook link, or other implementation pointer appearing in Paper direct links, the dataset description, or paper reproducibility context as the highest-priority evidence source.
- Inspect those paper-provided links before broad web search whenever they are present.
- For GitHub links, first use `inspect_github_repo`, `search_github_repo_paths`, `fetch_github_file`, and `list_github_release_assets` to inspect repository metadata, README files, candidate data paths, config files, scripts, notebooks, and release assets before falling back to generic page scraping or broad search.
- If a public GitHub repository link is discovered later through `search_download_evidence`, `fetch_page_text`, or `fetch_page_rendered`, treat it the same way as a paper-provided GitHub link and switch to the GitHub tools immediately.

Your objective
1. Read the previous execution log and previous review first.
2. If they contain concrete errors or reviewer concerns, fix those specific problems rather than rewriting blindly.
3. Use public evidence tools only as needed to confirm a real, reproducible acquisition path.
4. Download the real dataset used in the paper as closely as possible: provider/source, files or tables, asset universe, date range, frequency, labels or targets, features, preprocessing clues, and any documented filtering or transformations.
5. If you cannot support a safe script with evidence-backed URLs or API procedures, return `status="insufficient_evidence"` and `code=""`.

Tool workflow
Use tools in this order unless there is a strong reason not to:
1. If the provided paper context already contains a public GitHub repository link, start with `inspect_github_repo`.
2. For GitHub repositories already present in the paper context or discovered later during tool use, use `search_github_repo_paths`, `fetch_github_file`, `list_github_repo_tree`, and `list_github_release_assets` before broadening the search.
3. If the provided paper context already contains non-GitHub direct links or landing pages, use `fetch_page_text` on those URLs.
4. If the paper context, README, or any tool result cites a `drive.google.com/drive/folders/...` folder URL or a `drive.google.com/file/d/...` file URL, use `list_gdrive_folder` to enumerate per-file IDs, names, and MIME types before writing code. The folder URL itself and each file ID returned by `list_gdrive_folder` count as evidenced URLs for any subsequent `gdown.download(id=...)` or `gdown.download_folder(url=...)` call in the generated script. Do not guess Google Drive file IDs or `uc?export=download` URLs — always enumerate first.
5. Use `search_download_evidence` only to fill gaps, confirm ambiguous providers, or find official mirrors, repositories, or API docs that were not explicit in the paper context.
6. `fetch_page_text` on the most promising landing pages.
7. `probe_download_url` on candidate direct links before writing code.
8. `fetch_page_rendered` only if the plain page still looks JS-gated or ambiguous, for example when the page appears to rely on client-side rendering, the plain fetch hides download links, or the fetched text is clearly incomplete.
9. If `fetch_page_rendered` reveals candidate direct artifact links, run `probe_download_url` on those rendered-page candidates before using them in code.

Evidence rules
- Base the download path strictly on the dataset description, previous failures, previous review suggestions, and tool-fetched evidence.
- Prefer downloading directly from the original source named or implied by the paper whenever that source is publicly accessible and scriptable.
- If the paper's original source is unavailable, access-restricted, unstable, or otherwise unusable for reproducible scripting, then look for a credible alternative distribution of the same dataset, such as an official mirror, Kaggle, Hugging Face, or the paper's public repository.
- For standard public-market series — prices, returns, volumes, OHLCV for publicly traded instruments (equities, indices, futures, FX, crypto, rates, commodities) or standard macro indicators — the canonical values are vendor-independent. If the paper names a paid or terminal vendor (e.g. Bloomberg, Refinitiv, WRDS, CRSP, MetaTrader, ricequant, JoinQuant, Pinnacle CLC), use a free, scriptable, no-interactive-login vendor that delivers the same instrument identifier(s) at the same date range (or strict subset) and the same or higher frequency. Representative free families include yfinance for global equities/indices/FX/futures front-month, public exchange REST or ccxt for crypto, akshare/tushare/baostock for Chinese markets, FRED/IMF/World Bank for macro, and Frankfurter/exchangerate.host for FX; this list is illustrative, not exhaustive. Document the substitution in `reason`, including any methodology differences that may bias the data (e.g. split/dividend adjustment conventions, continuous-contract roll methods, mid vs trade prices, liquidity-provider differences). Do NOT substitute by changing the universe (different instruments), the period (different years), the asset class, or the granularity (e.g. tick to daily). Do NOT substitute when the dataset content is intrinsically paper-specific, such as hand-curated labels, proprietary signals, full-depth LOB feeds, or vendor-licensed historical constituents that the paper does not provide.
- Prefer the paper's original source over third-party mirrors whenever it is available.
- Prefer direct links extracted from `fetch_page_text` / `fetch_page_rendered` over guessed filenames or guessed repository layouts.
- Prefer GitHub-derived file paths, README instructions, and release assets from `inspect_github_repo`, `search_github_repo_paths`, `fetch_github_file`, `list_github_repo_tree`, and `list_github_release_assets` over guessing repository layout or artifact names.
- `probe_download_url` is the URL-validation step: use it to confirm redirects, status code, content type, content disposition, and whether a candidate looks like a real downloadable file versus an HTML landing page.
- Every external URL used in the generated code must also appear in the `evidence` array.
- In each evidence item, the `note` must say which tool found or validated that URL. For GitHub-derived evidence, cite the specific GitHub tool name.
- If you use an alternative source instead of the original paper source, explain in `reason` why the original source was not usable and why the alternative is an acceptable substitute.
- After rendered-page inspection, do not write code against a candidate download link until that concrete URL has been validated with `probe_download_url`.
- The `probe_download_url` validation requirement applies ONLY to direct artifact URLs and direct API endpoints that your script will hit itself (raw HTTPS GET, raw REST call, raw file URL). It does NOT apply to wrapper packages such as `yfinance`, `fredapi`, `ccxt`, `akshare`, `tushare`, `baostock`, `yahooquery`, `pandas-datareader`, `kaggle`, `huggingface_hub`, `pyarrow`, `tushare`, `wbdata`, and similar free or free-tier-API libraries that are obviously appropriate for the evidenced provider. These wrappers handle authentication, session cookies, request signing, redirects, alternate endpoints, and rate limits internally, so the underlying vendor URL is often not the URL the wrapper actually hits at runtime. A wrapper can work correctly even when the raw vendor endpoint returns HTTP 401, 403, 404, an HTML interstitial, a cookie/login wall, or a "page moved" redirect when probed directly. For wrapper packages, the wrapper's own official documentation (PyPI page, ReadTheDocs page, or GitHub README), validated with `fetch_page_text`, is sufficient evidence of a scriptable acquisition path. Do NOT downgrade `status` to `insufficient_evidence` solely because probing the bare vendor URL failed when an evidenced wrapper for that provider exists.
- The wrapper-package allowance is not a license to invent or guess packages. Only use packages that are well-established, obviously appropriate for the evidenced provider, and either listed in this prompt or evidenced via `fetch_page_text` against their official docs page in this round.
- If both the plain-fetch path and the rendered-page path still fail to reveal a concrete public scriptable path, AND no evidenced wrapper covers the provider, stop and return `insufficient_evidence`.
- If the paper only names a dataset vaguely and you cannot identify the exact real acquisition path without guessing, stop and return `insufficient_evidence`.

Script requirements
- The script runs in the same Python environment as fin-pipeline.
- Use only the Python standard library plus packages already installed in the fin-pipeline conda environment.
- Do not invent or guess package names. Prefer well-established libraries (`requests`, `pandas`, `numpy`) and provider SDKs that are obviously appropriate for the evidenced source (for example `kaggle` for Kaggle, `huggingface_hub` for Hugging Face, `yfinance` for Yahoo Finance, `fredapi` for FRED, `boto3` for S3).
- Do NOT invoke `pip install`, `pip3 install`, `python -m pip install`, `pip._internal`, `conda install`, `mamba install`, `apt`/`apt-get install`, `brew install`, `npm install`, `yum`/`dnf install`, or any other package manager at runtime — neither via `subprocess`, `os.system`, `os.popen`, a shell-string call, nor a user-defined wrapper that forwards an argv list to subprocess. The conda env is the source of truth for what is importable; runtime installs are statically rejected by the Phase 2 sandbox preflight and produce a `runtime_package_install` failure category.
- If a prior round's `Observed failure categories` list contains `module_not_found`, that means the previous script imported a package that is not installed; switch to a different acquisition path that uses the standard library or an alternative package, do not retry the same import.
- If a prior round's `Observed failure categories` list contains `runtime_package_install`, the previous script tried to install a package at runtime (banned per the rule above); replace that approach with one that uses only the conda env's existing packages, or return `insufficient_evidence` if no such path exists.
- Support `DRY_RUN`:
  - when `DRY_RUN=1`, do not download
  - instead print the planned steps, target URLs or providers, expected output paths, and any required environment variables
- On a real run, create `./data` and download at least one real data artifact.
- HTML scraping, paginated APIs, retries, checksum or file-size validation, Kaggle downloads, and Hugging Face downloads are allowed only when justified by the evidence and implemented safely.
- For public Google Drive content, prefer `gdown.download(id=<file_id>, output=<path>, quiet=True)` for single files and `gdown.download_folder(url=<folder_url>, output=<dir>, quiet=True)` for whole folders. Pass file IDs that were returned by `list_gdrive_folder`, or a folder URL that the paper or README explicitly cites. Do not call `gdown` against guessed IDs or `uc?export=download` URLs.

Authentication and credentials
- Public API keys or tokens that do not require interactive login are allowed, but they must be read from environment variables.
- Available environment variables: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, `FRED_API_KEY`.
- FRED guidance: prefer the official static download path when the evidence shows one; otherwise use `FRED_API_KEY` for API-based downloads.
- Google Drive: `gdown` works against public folders and files without credentials, so no env var is required. Do NOT use OAuth flows, service-account JSON files, or any login-required Drive access — those count as forbidden interactive auth.
- If a required credential is missing, fail clearly with a helpful message and a non-zero exit code.

Forbidden or invalid approaches
- Do not require manual browser login, interactive prompts, copy-paste steps, or cloud-compute workflows.
- Reject browser-login-only flows, CAPTCHA pages, and ephemeral signed URLs unless a stable documented API or direct official artifact link is also evidenced.
- Do not use privileged paths.
- Do not invent or guess URLs, file names, API parameters, login steps, bucket names, or repository layouts.
- Do not hardcode secrets.
- Do not fabricate data, generate placeholder artifacts, or claim success without a real evidence-backed acquisition path.

Output contract
Return JSON with exactly these keys:
- `evidence`: array of objects with keys `source_type`, `title`, `url`, `note`. Lists every URL or repository that supports the script you are about to write, including which tool found/validated each one.
- `code`: Python source code that uses only URLs from the `evidence` array above. Must be empty when the evidence is insufficient.
- `downloaded_dataset_description`: string; when the script is materializable, describe the concrete dataset the script will actually materialize in as much detail as possible. Include files, tables, assets, date ranges, frequencies, features, labels, preprocessing or filtering steps, transformations, and any deviations or narrowing relative to the paper description. Be explicit about three things downstream review will check: (1) which target / label / price column supports task evaluation; (2) any identity narrowing — if the public release covers a single bundle for what the paper benchmarks across multiple instruments / markets / regions, say so; (3) any auxiliary substitution — current vs point-in-time universe, free vs paid classification taxonomy, computed vs vendor-prepared feature columns. The reviewer will approve identity-narrowing or auxiliary-substitution payloads when this description is honest about them, and will reject when the description overstates coverage. Use the empty string when the evidence is insufficient.
- `reason`: short explanation of whether the evidence above is sufficient for a safe scriptable acquisition path, naming the decisive evidence or gap.
- `status`: one of `"ready"` or `"insufficient_evidence"`. Must be consistent with `evidence`/`code`/`reason`. Use `"insufficient_evidence"` only when `code` is empty.
""".strip()


def build_write_synthetic_dataset_code_prompt(
    dataset_name: str,
    dataset_description: str,
    aliases: list[str],
    paper_links: list[str] | None = None,
    reproducibility_context: str = "",
    execution_log: str = "",
    previous_review: str = "",
    previous_script: str = "",
    previous_evidence: list[dict[str, object]] | None = None,
    failure_categories: list[str] | None = None,
    timeout_sec: int | None = None,
) -> str:
    aliases_str = ", ".join(aliases) if aliases else "(none)"
    desc_block = (
        _truncate((dataset_description or "").strip(), field_name="dataset_description")
        or "(none)"
    )
    reproducibility_block = (
        _truncate(
            (reproducibility_context or "").strip(),
            field_name="reproducibility_context",
        )
        or "(none)"
    )
    paper_links_block = _truncate(
        _paper_links_block(paper_links),
        field_name="paper_links",
    )
    log_block = (
        _truncate(execution_log.strip(), keep_tail=True, field_name="execution_log")
        or "(none)"
    )
    review_block = (
        _truncate(previous_review.strip(), field_name="previous_review") or "(none)"
    )
    prev_script_block = (
        _truncate(previous_script.strip(), field_name="previous_script") or "(none)"
    )
    evidence_block = _truncate(
        json.dumps(previous_evidence or [], ensure_ascii=False, indent=2),
        field_name="previous_evidence",
    ) or "[]"
    failure_block = ", ".join(failure_categories or []) or "(none)"
    budget_block = _wall_clock_budget_block(timeout_sec)
    return f"""
Task: write a Python script that materializes the target synthetic dataset locally from the paper specification, or explicitly conclude that the specification is insufficient.

Context
- Dataset name: {dataset_name}
- Dataset kind: synthetic
- Dataset description: {desc_block}
- Paper reproducibility context: {reproducibility_block}
- Paper direct links:
{paper_links_block}
- Aliases: {aliases_str}
- Previous round script:
{prev_script_block}

- Previous execution log:
{log_block}

- Previous round review:
{review_block}

- Accumulated evidence from all prior rounds:
{evidence_block}

- Observed failure categories from prior rounds:
{failure_block}
{budget_block}

Track
- This is the synthetic-data generation track.
- The goal is to reproduce the synthetic dataset used in the paper's experiments as closely as the paper specification supports.
- If the synthetic workflow depends on real seed data, acquire only the seed data explicitly anchored by the paper, then perform the supported local generation steps.

Paper-provided links and implementation pointers
- Treat any GitHub link, dataset landing page, download URL, repository link, appendix code pointer, notebook link, or other implementation pointer appearing in Paper direct links, the dataset description, or paper reproducibility context as the highest-priority evidence source.
- Inspect those paper-provided links before broad web search whenever they are present.
- For GitHub links, first use `inspect_github_repo`, `search_github_repo_paths`, `fetch_github_file`, and `list_github_release_assets` to inspect repository metadata, README files, candidate notebooks, generation scripts, config files, seed-data references, and release assets before falling back to generic page scraping or broad search.
- If a public GitHub repository link is discovered later through `search_download_evidence`, `fetch_page_text`, or `fetch_page_rendered`, treat it the same way as a paper-provided GitHub link and switch to the GitHub tools immediately.

Your objective
1. Use only the paper-supported synthetic data specification provided above.
2. Repair prior round problems before rewriting from scratch.
3. Use public evidence tools when needed to discover or validate any real seed-data source mentioned by the paper.
4. Generate a deterministic local materialization script only when the specification is concrete enough.
5. Write the synthetic dataset used in the paper as closely as possible: generation process, distributions, parameters, horizons, sample counts, features, calibration steps, and any required seed-data transformation steps.
6. If the paper does not provide enough synthesis detail, or a required real seed-data source/access path is too underspecified, return `status="insufficient_evidence"` and `code=""`.

Tool workflow
Use tools in this order unless there is a strong reason not to:
1. If the provided paper context already contains a public GitHub repository link, start with `inspect_github_repo`.
2. For GitHub repositories already present in the paper context or discovered later during tool use, use `search_github_repo_paths`, `fetch_github_file`, `list_github_repo_tree`, and `list_github_release_assets` before broadening the search.
3. If the provided paper context already contains non-GitHub direct links or landing pages, use `fetch_page_text` on those URLs.
4. If the paper context, README, or any tool result cites a `drive.google.com/drive/folders/...` folder URL or a `drive.google.com/file/d/...` file URL for the seed data, use `list_gdrive_folder` to enumerate per-file IDs, names, and MIME types before writing code. The folder URL itself and each file ID returned by `list_gdrive_folder` count as evidenced URLs for any subsequent `gdown.download(id=...)` or `gdown.download_folder(url=...)` call in the generated script. Do not guess Google Drive file IDs or `uc?export=download` URLs — always enumerate first.
5. Use `search_download_evidence` only to fill gaps, find official sources, or confirm ambiguous appendix code, notebook references, or seed-data providers.
6. `fetch_page_text` on the most promising seed-data landing pages or repositories.
7. `probe_download_url` on candidate direct seed-data artifact URLs before writing code.
8. `fetch_page_rendered` only if the plain page still looks JS-gated or incomplete.
9. If `fetch_page_rendered` reveals candidate direct seed-data artifact URLs, probe those URLs before using them in code.

Seed-data acquisition rules
- If the paper specifies real seed data and gives enough source/access detail, the script may first acquire that seed data and then generate the synthetic dataset from it.
- Prefer acquiring seed data directly from the original source named or implied by the paper whenever that source is publicly accessible and scriptable.
- If the original seed-data source is unavailable, access-restricted, unstable, or otherwise unusable for reproducible scripting, then look for a credible alternative distribution of the same seed data, such as an official mirror, Kaggle, Hugging Face, or the paper's public repository.
- When the seed data is a standard public-market series (prices, returns, volumes, OHLCV for publicly traded instruments, or standard macro indicators), the canonical values are vendor-independent. If the paper names a paid or terminal vendor (e.g. Bloomberg, Refinitiv, WRDS, CRSP, MetaTrader, ricequant, JoinQuant, Pinnacle CLC), use a free, scriptable, no-interactive-login vendor that delivers the same instrument identifier(s) at the same date range (or strict subset) and the same or higher frequency. Representative free families include yfinance, public exchange REST or ccxt for crypto, akshare/tushare/baostock for Chinese markets, FRED/IMF/World Bank for macro, and Frankfurter/exchangerate.host for FX; this list is illustrative, not exhaustive. Document the substitution in `reason` with any methodology differences (split/dividend adjustment, continuous-contract roll, mid vs trade price, liquidity-provider differences). Do NOT substitute by changing the universe, period, asset class, or granularity. Do NOT substitute when the seed-data content is intrinsically paper-specific.
- Prefer the paper's original seed-data source over third-party mirrors whenever it is available.
- Use tools only to discover or validate seed-data sources that are anchored by the paper description.
- Every external URL used for seed-data acquisition must also appear in the structured `evidence` array.
- In each evidence item, the `note` must say which tool found or validated that URL.
- If you use an alternative seed-data source instead of the original paper source, explain in `reason` why the original source was not usable and why the alternative is an acceptable substitute.
- Do not guess seed-data providers, file names, API parameters, or repository layouts.
- If the synthetic procedure depends on real seed data but the source/access path is underspecified, return `insufficient_evidence`.

Materialization rules
- The script must create `./data` and write one or more real local artifacts there.
- Support `DRY_RUN`:
  - when `DRY_RUN=1`, do not generate files
  - instead print the generation plan, key parameters, output paths, and any assumptions
- Prefer plain files such as `.csv`, `.json`, `.parquet`, `.npy`, or a small README/metadata file describing the generated dataset.
- The script may use only the standard library plus already installed packages available in the fin-pipeline conda environment.
- Do not invent or guess package names. If a prior round's `Observed failure categories` list contains `module_not_found`, the previous script imported a package that is not installed; choose a different approach using the standard library or an alternative package rather than retrying the same import.
- Do NOT invoke `pip install`, `pip3 install`, `python -m pip install`, `pip._internal`, `conda install`, `mamba install`, `apt`/`apt-get install`, `brew install`, `npm install`, `yum`/`dnf install`, or any other package manager at runtime — neither via `subprocess`, `os.system`, `os.popen`, a shell-string call, nor a user-defined wrapper that forwards an argv list to subprocess. Runtime installs are statically rejected by the Phase 2 sandbox preflight and produce a `runtime_package_install` failure category. If a prior round's `Observed failure categories` list contains `runtime_package_install`, replace the runtime-install approach with one that uses only the conda env's existing packages, or return `insufficient_evidence`.
- For public Google Drive seed data, prefer `gdown.download(id=<file_id>, output=<path>, quiet=True)` for single files and `gdown.download_folder(url=<folder_url>, output=<dir>, quiet=True)` for whole folders. Pass file IDs that were returned by `list_gdrive_folder`, or a folder URL that the paper or README explicitly cites. Do not call `gdown` against guessed IDs or `uc?export=download` URLs. Do NOT use OAuth flows or service-account JSON for Drive access.
- Every synthesis step must be directly supported by the provided specification. Do not invent distributions, parameters, calibration targets, or sampling rules that are not evidenced.
- If both seed-data acquisition and synthetic generation are sufficiently specified, the script may do both in one workflow.
- Include comments only where needed to clarify a non-obvious generation step.

Forbidden or invalid approaches
- Do not fetch unrelated external data unless the provided specification explicitly requires a concrete real seed-data source and gives enough detail to reproduce it safely.
- Do not fabricate unsupported parameters or fill in missing equations with guesses.
- Do not emit placeholder files that do not reflect the described dataset.
- Do not claim success unless the script writes actual artifacts under `./data`.

Insufficient-evidence stop conditions
- Return `insufficient_evidence` if the paper requires real seed data but does not provide enough source/access detail to identify a safe reproducible acquisition path.
- Return `insufficient_evidence` if the paper does not provide enough synthesis details to implement the generator defensibly.
- Return `insufficient_evidence` rather than guessing missing parameters, calibration logic, or seed-data provenance.

Output contract
Return JSON with exactly these keys:
- `evidence`: array of objects with keys `source_type`, `title`, `url`, `note`. Lists every seed-data URL or repository that supports the script you are about to write, including which tool found/validated each one.
- `code`: Python source code that uses only seed-data URLs from the `evidence` array above. Must be empty when the synthetic specification is insufficient.
- `downloaded_dataset_description`: string; when the script is materializable, describe the concrete dataset the script will actually materialize in as much detail as possible. Include generated files, shapes, sample counts, features, parameters, processing steps, seed-data transformations, and any deviations or narrowing relative to the paper description. Use the empty string when the evidence is insufficient.
- `reason`: short explanation of whether the synthetic specification (and any required seed-data evidence) is sufficient, naming the decisive evidence or gap.
- `status`: one of `"ready"` or `"insufficient_evidence"`. Must be consistent with `evidence`/`code`/`reason`. Use `"insufficient_evidence"` only when `code` is empty.
""".strip()


def dataset_download_generation_schema() -> dict:
    return {
        "title": "DatasetAcquisitionCode",
        "description": "Output schema for dataset acquisition scripts (real download or synthetic generation) with evidence tracking.",
        "type": "object",
        "properties": {
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_type": {"type": "string"},
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["source_type", "title", "url", "note"],
                    "additionalProperties": False,
                },
            },
            "code": {"type": "string"},
            "downloaded_dataset_description": {"type": "string"},
            "reason": {"type": "string"},
            "status": {"type": "string", "enum": ["ready", "insufficient_evidence"]},
        },
        "required": ["evidence", "code", "downloaded_dataset_description", "reason", "status"],
        "additionalProperties": False,
    }


def build_download_review_prompt(
    dataset_name: str,
    dataset_description: str,
    dataset_kind: str,
    script: str,
    execution_log: str,
    evidence: list[dict[str, str]] | None = None,
    generation_status: str = "",
    generation_reason: str = "",
    timeout_sec: int | None = None,
) -> str:
    if str(dataset_kind or "").strip().lower() == "synthetic":
        return build_synthetic_dataset_review_prompt(
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            script=script,
            execution_log=execution_log,
            evidence=evidence,
            generation_status=generation_status,
            generation_reason=generation_reason,
            timeout_sec=timeout_sec,
        )
    return build_real_dataset_review_prompt(
        dataset_name=dataset_name,
        dataset_description=dataset_description,
        script=script,
        execution_log=execution_log,
        evidence=evidence,
        generation_status=generation_status,
        generation_reason=generation_reason,
        timeout_sec=timeout_sec,
    )


def build_real_dataset_review_prompt(
    dataset_name: str,
    dataset_description: str,
    script: str,
    execution_log: str,
    evidence: list[dict[str, str]] | None = None,
    generation_status: str = "",
    generation_reason: str = "",
    timeout_sec: int | None = None,
) -> str:
    evidence_raw = json.dumps(evidence or [], ensure_ascii=False, indent=2)
    desc_block = (
        _truncate((dataset_description or "").strip(), field_name="dataset_description")
        or "(none)"
    )
    evidence_block = _truncate(evidence_raw, field_name="evidence")
    script_block = _truncate(script.strip(), field_name="script") or "(none)"
    log_block = (
        _truncate(execution_log.strip(), keep_tail=True, field_name="execution_log")
        or "(none)"
    )
    generation_status_block = generation_status.strip() or "(none)"
    generation_reason_block = generation_reason.strip() or "(none)"
    budget_block = _wall_clock_budget_block(timeout_sec)
    budget_review_bullet = _wall_clock_budget_review_bullet(
        timeout_sec, materialization_label="acquisition"
    )
    return f"""
You are reviewing a real-dataset acquisition script for local execution.

Dataset name:
{dataset_name}

Dataset description:
{desc_block}

Evidence used by the code generator:
{evidence_block}

Current code generation status:
{generation_status_block}

Current code generation reason:
{generation_reason_block}

Current acquisition code:
{script_block}

Current acquisition execution log:
{log_block}
{budget_block}

Review criteria:
- local safety, including no hardcoded secrets, no interactive login requirements, no privileged paths, and no destructive behavior outside the dataset working directory
- reproducibility, including clear handling of required environment variables, deterministic output paths under ./data, and clear failure behavior when credentials or upstream files are missing
- diagnosis quality: identify the concrete reason this round failed or was marked insufficient, including why the code failed, why the download failed, or why there is insufficient evidence for a safe scriptable download path
- correctness of the intended real-data acquisition path: whether the script exactly matches the dataset description, whether the evidence block supports the intended provider or download path, and whether the current generation reason is valid
- dataset-track correctness: the script must implement a real acquisition path and must reject fake, sampled, synthetic, or placeholder stand-ins
- runtime robustness: whether the script handles common HTTP and file failure modes sensibly, uses retries only when reasonable, handles pagination or HTML parsing safely when needed, and validates downloaded files when the source provides enough clues
- allowed download patterns such as HTML scraping, paginated APIs, retries, checksum or file-size validation, Kaggle access, and Hugging Face access are acceptable when justified by the evidence and implemented safely
- authentication behavior: environment-variable-based API keys or tokens are acceptable, with no hardcoded secrets and no interactive login requirements; browser-based auth, manual copy-paste steps, or interactive prompts must be rejected
- every external URL in the script must be supported by the evidence block
- GitHub-derived README guidance, repo paths, and release assets are acceptable evidence only when they are cited through the GitHub evidence tools and the script uses the same URLs or artifact paths
- the script should write real artifacts under ./data and should not fake downloads, silently succeed without producing data, or emit misleading success messages when the download actually failed
- any fake download, placeholder artifact, guessed URL, hardcoded secret, unsupported dependency assumption, or interactive auth flow must be rejected
- if generation_status is not "ready", treat the generation_reason and execution log as first-class evidence and explain what additional evidence, code changes, or search strategy would be needed for the next round
{budget_review_bullet}
- TASK COMPLETENESS (apply the strictest tier that fits):
  - (a) Missing target source → REJECT (`task_incomplete_partial_acquisition`). The target is whichever artifact the paper scores its model against — explicit label, price column, return series, reference sample, ground-truth signal, expert annotation, classification tag, or any column from which one of these can be derived deterministically. If none of those is in the payload, the task has no measurable outcome and the script must be rejected. The most common failure shape is a paper that combines multiple modalities or sources — text + price, news + returns, transcripts + market data, audio + returns, factor signals + asset returns, options chain + spot price, satellite imagery + economic indicator — where the script fetched only the obvious primary source and silently omitted the partner source the model is scored against. The correct response is to extend the acquisition path so the missing target source is included from a publicly scriptable provider (a free wrapper such as `yfinance`, `fredapi`, `akshare`, `baostock`, `ccxt`, etc. usually fills standard market-data, macro, or crypto gaps), or to mark the dataset as not-acquirable when no public path exists. Do NOT fabricate, sample, or substitute placeholder labels.
  - (b) Bundle contains what the task needs → APPROVE, even if the bundle's identity does not 1:1 match the candidate name. The check is "is the data we need present?", not "is the bundle named like the candidate?". Accept generic filenames, supersets, subsets, aggregated splits, differently-scoped releases, and paper-authored bundles whose internal layout does not document the candidate's narrow name — provided target derivation per (a) holds. Downstream consumers decide per-task suitability. Required: `downloaded_dataset_description` honestly records what's actually inside (coverage, naming, what is or is not provable about identity).
  - (c) Auxiliary component missing or substituted → APPROVE with caveats. Auxiliary = point-in-time universe history, classification taxonomies, vendor-specific feature engineering, alternative-data side channels, input features that can be deterministically computed from the payload. Required: every substitution named explicitly in `downloaded_dataset_description`.
  - Cross-cutting: misrepresenting the payload in `downloaded_dataset_description` (silent substitution, overstated coverage) is itself grounds for rejection (`manifest_overstates_payload`).

Respond with JSON containing these fields in order:

1. analysis (string): Think step-by-step before deciding. Your analysis MUST cover:
   - Safety: Does the script avoid arbitrary code execution, file system damage, or network abuse? No hardcoded secrets, no interactive login, no privileged paths?
   - Correctness: Does it implement the intended real-data acquisition steps for {dataset_name}? Are providers, URLs, file formats, and extraction steps correct and supported by the evidence block?
   - Reproducibility: Will the script produce consistent results across runs? Are there race conditions, hardcoded temp paths, or missing error handling?
   - Execution log: What does the execution output show? Did it complete successfully or are there errors/warnings?
   - Evidence validity: Does every external URL in the script appear in the evidence block? Is the generation reason valid?
   - Task completeness (apply the three-tier rule above): classify each missing piece of the paper's task into (a) target-derivation, (b) identity-narrowing, or (c) auxiliary. Reject only when (a) applies — the payload contains neither the explicit target nor a column from which it can be derived deterministically, and the missing source cannot be added from a publicly scriptable provider. For (b) and (c), approve provided the `downloaded_dataset_description` honestly records what was actually acquired, what is missing, and any substitutions made.
   - Manifest honesty: regardless of tier, if `downloaded_dataset_description` overstates the payload's identity, coverage, or feature completeness relative to what the script actually fetches, treat that as grounds for rejection.
   - Previous-round fixes: If there was a prior round, have the identified issues been resolved?
   - Generation status: If generation_status is not "ready", explain what additional evidence or changes are needed for the next round.

2. approved (boolean): true only if the script is safe, correct, evidence-backed, the execution log indicates a successful or clearly-successor-ready local download path, AND the acquired payload passes the three-tier task-completeness rule (no missing target-derivation; identity-narrowing or auxiliary gaps are honestly recorded in `downloaded_dataset_description`). If approved is true, issues and suggestions should be empty arrays.

3. issues (array of strings): Descriptions of ALL observed problems or risks. Empty array if approved is true. When tier (a) applies (missing target source that cannot be derived from the payload and cannot be added from a public scriptable provider), include an issue tagged "task_incomplete_partial_acquisition" with a concrete explanation of which target component is missing and what the paper required. When the manifest text overstates what was acquired (silent substitution, identity claim not supported by evidence), include an issue tagged "manifest_overstates_payload" with a concrete explanation.

4. suggestions (array of strings): Detailed, actionable hints for improving download success. Empty array if approved is true.
""".strip()


def build_synthetic_dataset_review_prompt(
    dataset_name: str,
    dataset_description: str,
    script: str,
    execution_log: str,
    evidence: list[dict[str, str]] | None = None,
    generation_status: str = "",
    generation_reason: str = "",
    timeout_sec: int | None = None,
) -> str:
    evidence_raw = json.dumps(evidence or [], ensure_ascii=False, indent=2)
    desc_block = (
        _truncate((dataset_description or "").strip(), field_name="dataset_description")
        or "(none)"
    )
    evidence_block = _truncate(evidence_raw, field_name="evidence")
    script_block = _truncate(script.strip(), field_name="script") or "(none)"
    log_block = (
        _truncate(execution_log.strip(), keep_tail=True, field_name="execution_log")
        or "(none)"
    )
    generation_status_block = generation_status.strip() or "(none)"
    generation_reason_block = generation_reason.strip() or "(none)"
    budget_block = _wall_clock_budget_block(timeout_sec)
    budget_review_bullet = _wall_clock_budget_review_bullet(
        timeout_sec, materialization_label="materialization"
    )
    return f"""
You are reviewing a synthetic-dataset materialization script for local execution.

Dataset name:
{dataset_name}

Dataset description:
{desc_block}

Evidence used by the code generator:
{evidence_block}

Current code generation status:
{generation_status_block}

Current code generation reason:
{generation_reason_block}

Current generation code:
{script_block}

Current generation execution log:
{log_block}
{budget_block}

Review criteria:
- local safety, including no hardcoded secrets, no interactive login requirements, no privileged paths, and no destructive behavior outside the dataset working directory
- reproducibility, including deterministic output paths under ./data, clear failure behavior, and explicit handling of any required environment variables for real seed-data acquisition
- diagnosis quality: identify the concrete reason this round failed or was marked insufficient, including why the code failed, why a required seed-data acquisition step failed, or why the synthetic specification is still insufficient
- correctness of the intended synthetic-data path: whether the script matches the dataset description and generation details, whether any claimed generation parameters are actually supported, and whether the current generation reason is valid
- dataset-track correctness: the script may synthesize local data artifacts only when that generation is directly supported by the provided synthetic specification
- seed-data correctness: if the synthetic workflow requires real seed data, every provider, URL, repository, or API path must be supported by the evidence block and must match the paper-supported seed-data description
- GitHub-derived README guidance, repo paths, notebooks, and release assets are acceptable evidence only when they are cited through the GitHub evidence tools and the script uses the same URLs or artifact paths
- runtime robustness: whether the script handles common HTTP and file failure modes sensibly for seed-data acquisition, validates generated outputs when possible, and fails clearly on underspecified or impossible steps
- authentication behavior: environment-variable-based API keys or tokens are acceptable for seed-data acquisition, with no hardcoded secrets and no interactive login requirements
- the script should write real artifacts under ./data and should not fake generation, silently succeed without producing data, or emit misleading success messages
- guessed distributions, guessed calibration targets, guessed sample construction logic, or placeholder artifacts must be rejected
- if generation_status is not "ready", treat the generation_reason and execution log as first-class evidence and explain what additional paper detail, source evidence, or code changes would be needed for the next round
{budget_review_bullet}

Respond with JSON containing these fields in order:

1. analysis (string): Think step-by-step before deciding. Your analysis MUST cover:
   - Safety: Does the script avoid arbitrary code execution, file system damage, or network abuse? No hardcoded secrets, no interactive login, no privileged paths?
   - Correctness: Does it reproduce the intended synthetic dataset for {dataset_name}? Are the generation steps, parameters, shapes, horizons, features, and any seed-data acquisition steps supported by the description and evidence block?
   - Reproducibility: Will the script produce consistent results across runs? Are seeds, output paths, and failure modes handled clearly?
   - Execution log: What does the execution output show? Did it complete successfully or are there errors or warnings?
   - Evidence validity: Does every external URL in the script appear in the evidence block? Is the generation reason valid?
   - Previous-round fixes: If there was a prior round, have the identified issues been resolved?
   - Generation status: If generation_status is not "ready", explain what additional evidence or changes are needed for the next round.

2. approved (boolean): true only if the script is safe, correctly reproduces the supported synthetic workflow, and the execution log indicates a successful or clearly-successor-ready local materialization path. If approved is true, issues and suggestions should be empty arrays.

3. issues (array of strings): Descriptions of ALL observed problems or risks. Empty array if approved is true.

4. suggestions (array of strings): Detailed, actionable hints for improving generation success. Empty array if approved is true.
""".strip()


def build_reward_reuse_prompt(
    *,
    candidate_name: str,
    candidate_aliases: list[str],
    candidate_description: str,
    existing_rewards: list[dict[str, str]],
) -> str:
    catalog_json = _truncate(
        json.dumps(existing_rewards, ensure_ascii=False), field_name="existing_rewards"
    )
    aliases = ", ".join(a for a in candidate_aliases if a) or "(none)"
    desc_block = (
        _truncate(
            (candidate_description or "").strip(), field_name="candidate_description"
        )
        or "(none)"
    )
    return f"""
You are deciding whether a newly extracted reward can reuse an already existing local reward.

Candidate reward:
- name: {candidate_name}
- aliases: {aliases}
- description: {desc_block}

Existing local rewards (name + description):
{catalog_json}

Decision policy:
1) reuse=true only when there is clear evidence of the same computation.
2) Strong positive signals include: exact same reward name, standard notation or spelling variants, or an obvious alias/rename for the same formula.
3) Reuse is not allowed when the candidate changes the threshold, horizon, aggregation rule, normalization, averaging mode, or any other part of the computation.
4) Do not reuse based on topical similarity alone.
5) If uncertain, choose reuse=false.
6) matched_name must be one of existing reward names when reuse=true.

Output JSON only:
- reason: string. Walk through the candidate's computation and compare it concretely against the closest existing rewards (formula, aggregation rule, normalization, averaging mode). If aggregation/averaging/stability differs (e.g. mean(IC)/std(IC) vs raw correlation), say so explicitly here; that conclusion must be reflected in `reuse` below.
- matched_name: string (empty when reuse=false)
- reuse: boolean. If `reason` says the computations differ in any of threshold, horizon, aggregation rule, normalization, or averaging mode, this MUST be false.
""".strip()


def reward_reuse_schema() -> dict:
    return {
        "title": "RewardReuseDecision",
        "description": "Determines whether a newly extracted metric can reuse an existing reward computation in the local library.",
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "matched_name": {"type": "string"},
            "reuse": {"type": "boolean"},
        },
        "required": ["reason", "matched_name", "reuse"],
        "additionalProperties": False,
    }


def build_write_reward_code_prompt(
    reward_name: str,
    aliases: list[str],
    description: str,
    execution_log: str = "",
    previous_review: str = "",
    previous_script: str = "",
) -> str:
    from openfinai_pipeline.prompts.contract_rules import REWARD_FN_CONTRACT_RULES

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )
    tmpl = env.get_template("reward_fn_prompt.j2")
    return tmpl.render(
        reward_name=reward_name,
        aliases=", ".join(a for a in aliases if a) or "(none)",
        description=_truncate(description.strip(), field_name="description")
        or "(none)",
        previous_script=_truncate(previous_script.strip(), field_name="previous_script")
        or "(none)",
        execution_log=_truncate(
            execution_log.strip(), keep_tail=True, field_name="execution_log"
        )
        or "(none)",
        previous_review=_truncate(previous_review.strip(), field_name="previous_review")
        or "(none)",
        contract_rules=REWARD_FN_CONTRACT_RULES,
    ).strip()


def code_generation_schema() -> dict:
    """JSON schema for the reward-codegen LLM output.

    Under the canonical-name contract the LLM emits only the source code;
    parameter names ARE the wiring and the runtime introspects them.
    """
    return {
        "title": "EvaluationCode",
        "description": (
            "PyTorch Loss subclass source. Param names follow the "
            "canonical-name contract (R1-R6); no JSON envelope."
        ),
        "type": "object",
        "properties": {
            "code": {"type": "string"},
        },
        "required": ["code"],
        "additionalProperties": False,
    }


def build_reward_review_prompt(
    reward_code: str,
    reward_name: str,
    reward_description: str,
    execution_log: str = "",
    previous_script: str = "",
    previous_execution_log: str = "",
) -> str:
    from openfinai_pipeline.prompts.contract_rules import REWARD_FN_CONTRACT_RULES

    desc_block = (
        _truncate((reward_description or "").strip(), field_name="reward_description")
        or "(none)"
    )
    prev_script_block = (
        _truncate(previous_script.strip(), field_name="previous_script") or "(none)"
    )
    prev_log_block = (
        _truncate(
            previous_execution_log.strip(),
            keep_tail=True,
            field_name="previous_execution_log",
        )
        or "(none)"
    )
    exec_block = (
        _truncate(execution_log.strip(), keep_tail=True, field_name="execution_log")
        or "(none)"
    )
    reward_code_block = _truncate(reward_code.strip(), field_name="reward_code") or "(none)"
    return f"""
You are reviewing a PyTorch evaluation reward module that should define
exactly one Loss subclass (imported from reward_bank).

{REWARD_FN_CONTRACT_RULES}

Reward expected:
{reward_name}

Reward description:
{desc_block}

Previous round script:
{prev_script_block}

Previous round execution log:
{prev_log_block}

Current Reward code for review:
{reward_code_block}

Current Reward execution log (AST contract check + pytest results):
{exec_block}

Respond with JSON containing these fields in order:

1. analysis (string): Think step-by-step before deciding. Your analysis should consider:
   - Contract compliance (RULES R1-R6): canonical names with no defaults; non-canonical names with literal defaults; init-only vs forward-only canonical name asymmetry; one Loss subclass; torch only; no separate compute().
   - **Name precision (Layer 1 enforcement):** any forward/init parameter that uses a vague generic name (`output`, `value`, `result`, `data`, `important_output`, `prediction`, `target` — anything that describes a ROLE rather than the SEMANTIC CONTENT) MUST be rejected. Suggest a precise canonical name (`gt`, `pred`, etc.) or, if the metric needs something not yet canonical, a precise descriptive name tied to the upstream semantic role (e.g. `volatility_forecast`, `regime_label`, `classification_logits`). Generic role names cause runtime collisions when multiple metrics live in the same evaluator.
   - Whether the implementation matches the mathematical definition in the description (formula, aggregation, normalization, thresholds).
   - Edge cases (empty / NaN / mismatched shapes) and numerical stability (divisions by zero, log of zero, missing epsilons).
   - API compliance: subclasses Loss, forward() returns a scalar torch.Tensor, torch ops only (no NumPy at runtime).
   - Test results — what passed, what failed, why.
   - If there was a prior round, whether the previously identified issues are now resolved.

   Common pitfalls to keep in mind (won't apply to every reward — use judgment):
   - When a description specifies an aggregation, averaging, or normalization step ("1/T sum", "ratio of mean to std", "top decile"), the corresponding step should appear in the code; a silently missing aggregation changes the semantics.
   - A metric whose result is independent of the agent's predicted output (`pred` / `fake_emb` / `pred_samples`) is broken regardless of what else the math does. If the agent's input doesn't visibly influence the return value, REJECT.
   - Implicit shape / axis assumptions inside reshapes and reductions are a frequent source of degenerate scalars (NaN, single-point std, divide-by-zero); worth a sanity check whenever the code reshapes or iterates.

2. approved (boolean): true ONLY if RULES R1-R6 hold, name precision is satisfied, the implementation is mathematically correct, AND all tests pass.

3. issues (array of strings): Concise descriptions of any problems found.

4. suggestions (array of strings): Detailed, actionable fixes for each issue. CRITICAL: never suggest changes that violate RULES R1-R6. In particular, NEVER suggest moving ground-truth-like canonical names (`gt`, `real_emb`, `gt_samples`) into `forward()` — R3 forbids that and the AST validator will reject it. If your previous-round suggestion conflicted with R1-R6, treat that as the source of the failure and reverse it.
""".strip()


def review_schema() -> dict:
    return {
        "title": "CodeReview",
        "description": "Review assessment for generated code, including analysis, approval status, issues, and actionable suggestions.",
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "approved": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "suggestions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["analysis", "approved", "issues", "suggestions"],
        "additionalProperties": False,
    }


def build_label_dataset_prompt(
    *,
    name: str,
    description: str,
    downloaded_dataset_description: str,
    interaction_model_hint: str,
    payload_files: list[str],
    preview: dict,
    dataset_kind: str,
) -> str:
    """Build the post-download labeler prompt.

    The labeler sees ACTUAL landed files (via ``payload_files``) and their
    previews (shapes, column names, head lines), plus the pre-execution
    description prose. It assigns a ``role`` to every file. Whether the
    dataset has a separate held-out ground-truth artifact is implicit
    in the role tags (``ground_truth`` / ``reference``) — there is no
    separate ``has_ground_truth`` boolean for the LLM to author; the
    role tags ARE the answer.
    """
    desc_block = (
        _truncate((description or "").strip(), field_name="description") or "(none)"
    )
    downloaded_desc_block = (
        _truncate(
            (downloaded_dataset_description or "").strip(),
            field_name="downloaded_dataset_description",
        )
        or "(none)"
    )
    interaction_hint_block = (interaction_model_hint or "").strip() or "(none)"
    dataset_kind_block = (dataset_kind or "").strip() or "real"
    payload_files_block = (
        _truncate(
            json.dumps(list(payload_files), ensure_ascii=False, indent=2),
            field_name="payload_files",
        )
        or "[]"
    )
    preview_block = (
        _truncate(
            json.dumps(preview or {}, ensure_ascii=False, indent=2),
            field_name="preview",
        )
        or "{}"
    )
    roles_list = ", ".join(f"`{v}`" for v in ROLE_VOCAB)
    formats_list = ", ".join(f"`{v}`" for v in FORMAT_VOCAB)
    return f"""
You are a data-forensics assistant. A dataset has just been downloaded to
disk. Given the ACTUAL file listing and previews, assign a semantic role to
every file. The role tags determine whether the dataset ships a separate
held-out evaluation target (any file tagged ``ground_truth`` or
``reference`` means yes) or not — you do not need to author a separate
boolean for that.

Dataset identity
- Name: {name}
- Dataset kind: {dataset_kind_block}
- Paper description: {desc_block}
- Downloaded-dataset description (pre-execution claim from the downloader):
{downloaded_desc_block}

Intended use (hint only; defer to actual file contents)
- Interaction model hint: {interaction_hint_block}
- Hint values may include: `forecasting`, `generative`, `trading`,
  `realtime_forecasting`, `realtime_trading`.

Actual files on disk (relative to the data directory)
{payload_files_block}

File previews (per-file fields depend on format):
- `.npy` / `.npz`: `dtype`, `shape` (and per-member dtype/shape for npz)
- `.csv`: `preview_text` (first ~10 lines, truncated to ~1500 chars) plus
  structured fields read via pandas: `columns` (full ordered list),
  `dtypes` (per-column dtype string), `row_count_sampled` (rows used for
  the stats; capped — `row_count_truncated_at` flags the cap),
  `null_counts` (only columns with nonzero nulls; `null_counts_overflow`
  when capped), and `describe` (count/mean/std/min/25%/50%/75%/max for up
  to ~30 numeric columns; `describe_overflow` when capped). Stats may be
  absent when pandas can't parse the file — text head still present.
- `.txt` / `.json` / `.jsonl` / `.md`: `preview_text` — the first
  ~5 lines (truncated to ~600 chars)
- `.parquet`: `columns`, `dtypes`, `row_count`, `row_groups` — read
  deterministically from the file's self-describing schema
- `.feather` / `.arrow` / `.ipc`: `columns`, `dtypes`, `row_count` —
  read from the Arrow IPC schema
- `.h5` / `.hdf5`: `datasets` — list of `{{name, shape, dtype}}` for each
  HDF5 Dataset (capped, with `datasets_overflow` when truncated)
- Other extensions: only `path` and `size_bytes`
{preview_block}

Role vocabulary (pick exactly one per file)
- {roles_list}

Format vocabulary (canonical values)
- {formats_list}

Decision rules
1. If the dataset includes a SEPARATE file containing the held-out
   evaluation target (labels/targets/returns aligned 1:1 with a
   features file, kept in its own artifact), tag that file
   `role=ground_truth`. Common in forecasting/regression/classification
   datasets where the labels live in their own CSV/npy.
2. If the dataset is generative/distributional (``interaction_model_hint``
   is `generative`), and a separate REFERENCE sample set is present
   that the agent's generated samples will be compared against via
   distributional metrics (FID/KID/W1), tag it `role=reference`.
3. If the held-out target is NOT a separate file — for example,
   forecasting datasets where the target is a precomputed column in
   the features file, or where the loader is expected to derive the
   target by shifting a feature column, or where scoring comes from
   live-env rewards (trading/realtime_trading) or delayed labels
   (realtime_forecasting) — do NOT tag any artifact as `ground_truth`
   or `reference`. The Phase 4 loader will source the target itself
   from the feature columns and record its choice in
   ``ground_truth_provenance``. Tag every file as `features` or
   `metadata` as appropriate. This is a legitimate `labeled` outcome.
4. Tag input features as `role=features`.
5. Tag readmes, licenses, schema files, or other non-data artifacts as
   `role=metadata`.
6. Tag other files not used by either agent or evaluator as
   `role=auxiliary`.
7. If you truly cannot decide between candidate ground-truth files (e.g.
   multiple equally-plausible splits with no disambiguating signal),
   return `manifest_status="unresolved"` with `labeler_notes` explaining
   what information is missing. This will fail Phase 4 assembly loudly —
   use it only when classification is genuinely ambiguous, not as a
   default.

Guidance for `format`
- Use the file's actual on-disk format, not what the prose description
  claims. Preview dtypes and shapes tell you definitively whether a file
  is npy vs. npz. Column names + `.csv`/`.parquet` suffix disambiguate
  tabular formats. If unsure, choose the closest canonical value; use
  `other` only as a last resort.

Guidance for `shape` / `columns` / `dtype`
- Copy directly from the preview when available. Leave arrays empty when
  the preview did not include that info. Do not guess.

Output contract
Return JSON with exactly these keys:
- `labeler_notes`: short string; record any observations, ambiguities, or unusual findings here. If the target is a precomputed column inside a features file (rather than its own artifact), or if the loader will need to derive the target by shifting, name the column here so Phase 4's loader prompt has a hint to consider. Empty string is acceptable for an obviously-clean labeled outcome.
- `artifacts`: array of objects, one per file in the listing. Each entry
  MUST have `role`, `path`, `format`. Shape/columns/dtype/description are
  optional and should be copied from the preview when present.
- `manifest_status`: `"labeled"` if you assigned roles confidently, or `"unresolved"` if the choice is ambiguous (in which case `labeler_notes` should already have explained why).

Final question: is there a SEPARATE file on disk that holds the held-out
evaluation target? If yes, tag it `role=ground_truth` (or `role=reference`
for generative). If no — including the case where the target is embedded
as a column in a features file — leave every artifact tagged `features` /
`metadata` / `auxiliary` and let Phase 4's loader source the target
itself.
""".strip()
