import logging
import re
from collections import Counter
from typing import Any

from openfinai_pipeline.corpus.aggregator import (
    DatasetCandidate,
    RewardCandidate,
)
from openfinai_pipeline.corpus.download_tools import (
    build_download_evidence_tools,
)
from openfinai_pipeline.corpus.manifest import (
    ROLE_VOCAB,
    canonicalize_format,
    infer_format_from_suffix,
)
from openfinai_pipeline.llm import LLMService
from openfinai_pipeline.prompts.corpus_builder import (
    build_label_dataset_prompt,
    build_write_synthetic_dataset_code_prompt,
    build_write_download_code_prompt,
    build_write_reward_code_prompt,
    code_generation_schema,
    dataset_download_generation_schema,
    dataset_label_schema,
)

logger = logging.getLogger(__name__)


def generate_download_code(
    candidate: DatasetCandidate,
    llm: LLMService | None,
    *,
    execution_log: str,
    previous_review: str,
    previous_script: str,
    previous_evidence: list[dict[str, Any]] | None = None,
    failure_categories: list[str] | None = None,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    if candidate.dataset_kind == "synthetic":
        return _generate_synthetic_dataset_code(
            candidate,
            llm,
            execution_log=execution_log,
            previous_review=previous_review,
            previous_script=previous_script,
            previous_evidence=previous_evidence,
            failure_categories=failure_categories,
            timeout_sec=timeout_sec,
        )
    shortcut = _provider_shortcut_download_payload(candidate)
    if shortcut is not None and not previous_script.strip():
        return shortcut
    if llm is None:
        return {
            "status": "failed_execution",
            "code": "",
            "downloaded_dataset_description": "",
            "evidence": [],
            "reason": "no_llm",
        }
    try:
        payload = llm.complete_structured_with_tools(
            build_write_download_code_prompt(
                dataset_name=candidate.name,
                dataset_description=candidate.description,
                dataset_kind=candidate.dataset_kind,
                paper_links=candidate.paper_links,
                reproducibility_context=candidate.reproducibility_context,
                aliases=candidate.aliases,
                execution_log=execution_log,
                previous_review=previous_review,
                previous_script=previous_script,
                previous_evidence=previous_evidence or [],
                failure_categories=failure_categories or [],
                timeout_sec=timeout_sec,
            ),
            dataset_download_generation_schema(),
            build_download_evidence_tools(),
        )
    except Exception as exc:
        logger.warning(
            "dataset download code generation failed scope=%s name=%s error=%s",
            candidate.scope_id,
            candidate.name,
            exc,
        )
        return {
            "status": "failed_execution",
            "code": "",
            "downloaded_dataset_description": "",
            "evidence": [],
            "reason": str(exc),
        }
    code = _normalize_code_block(str(payload.get("code", "")))
    status = str(payload.get("status", "")).strip() or "insufficient_evidence"
    if status != "ready":
        code = ""
    return {
        "status": status,
        "code": code,
        "downloaded_dataset_description": str(
            payload.get("downloaded_dataset_description", "")
        ).strip(),
        "evidence": payload.get("evidence", []) or [],
        "reason": str(payload.get("reason", "")).strip(),
    }


def _generate_synthetic_dataset_code(
    candidate: DatasetCandidate,
    llm: LLMService | None,
    *,
    execution_log: str,
    previous_review: str,
    previous_script: str,
    previous_evidence: list[dict[str, Any]] | None = None,
    failure_categories: list[str] | None = None,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    if llm is None:
        return {
            "status": "failed_execution",
            "code": "",
            "downloaded_dataset_description": "",
            "evidence": [],
            "reason": "no_llm",
        }
    try:
        payload = llm.complete_structured_with_tools(
            build_write_synthetic_dataset_code_prompt(
                dataset_name=candidate.name,
                dataset_description=candidate.description,
                paper_links=candidate.paper_links,
                reproducibility_context=candidate.reproducibility_context,
                aliases=candidate.aliases,
                execution_log=execution_log,
                previous_review=previous_review,
                previous_script=previous_script,
                previous_evidence=previous_evidence or [],
                failure_categories=failure_categories or [],
                timeout_sec=timeout_sec,
            ),
            dataset_download_generation_schema(),
            build_download_evidence_tools(),
        )
    except Exception as exc:
        logger.warning(
            "synthetic dataset code generation failed scope=%s name=%s error=%s",
            candidate.scope_id,
            candidate.name,
            exc,
        )
        return {
            "status": "failed_execution",
            "code": "",
            "downloaded_dataset_description": "",
            "evidence": [],
            "reason": str(exc),
        }
    code = _normalize_code_block(str(payload.get("code", "")))
    status = str(payload.get("status", "")).strip() or "insufficient_evidence"
    if status != "ready":
        code = ""
    evidence = payload.get("evidence", []) or []
    if not isinstance(evidence, list):
        evidence = []
    return {
        "status": status,
        "code": code,
        "downloaded_dataset_description": str(
            payload.get("downloaded_dataset_description", "")
        ).strip(),
        "evidence": evidence,
        "reason": str(payload.get("reason", "")).strip(),
    }


def generate_reward_code(
    reward: RewardCandidate,
    llm: Any | None,
    *,
    execution_log: str,
    previous_review: str,
    previous_script: str,
) -> dict[str, Any]:
    """Run one LLM round and return ``{"code": str}``.

    Under the canonical-name contract, parameter names ARE the wiring —
    the LLM only emits source code. The runtime evaluator introspects
    ``__init__`` / ``forward`` signatures to bind ctx tensors. The AST
    validator enforces R1-R6 before pytest runs.

    When the LLM is unavailable or returns an empty response, returns a dict
    with ``code=""``. Callers treat a missing ``code`` as the failure signal.
    """
    fallback: dict[str, Any] = {"code": ""}
    if llm is None:
        return fallback
    payload = llm.complete_or_fallback(
        build_write_reward_code_prompt(
            reward_name=reward.name,
            aliases=reward.aliases,
            description=reward.description,
            execution_log=execution_log,
            previous_review=previous_review,
            previous_script=previous_script,
        ),
        code_generation_schema(),
        fallback=dict(fallback),
    )
    code = _normalize_code_block(str(payload.get("code", "")))
    if not code:
        return fallback
    class_defs = re.findall(r"^class\s+\w+\(.*Loss.*\):", code, re.MULTILINE)
    if len(class_defs) > 1:
        logger.warning(
            "LLM generated %d Loss subclasses for reward '%s' "
            "(expected 1): %s",
            len(class_defs),
            reward.name,
            [c.split("(")[0].replace("class ", "") for c in class_defs],
        )
    return {"code": code}


def _normalize_code_block(text: str) -> str:
    code = text.strip()
    if code.startswith("```"):
        code = code.strip("`")
        if "\n" in code:
            code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code[:-3].rstrip()
    return code.strip()


def _provider_shortcut_download_payload(candidate: DatasetCandidate) -> dict[str, Any] | None:
    text = " ".join(
        [
            candidate.name or "",
            candidate.description or "",
            " ".join(candidate.aliases or []),
        ]
    )
    kaggle_match = re.search(r"https?://www\.kaggle\.com/datasets/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", text)
    if kaggle_match:
        owner, dataset = kaggle_match.groups()
        url = kaggle_match.group(0)
        code = f"""import os
import shutil
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> int:
    target_dir = Path("data")
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_ref = "{owner}/{dataset}"
    if os.getenv("DRY_RUN") == "1":
        print("provider=Kaggle")
        print(f"dataset={{dataset_ref}}")
        print(f"output_dir={{target_dir.resolve()}}")
        print("required_env=KAGGLE_USERNAME,KAGGLE_KEY")
        return 0
    missing = [key for key in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not os.getenv(key)]
    if missing:
        print(f"Missing required env vars: {{', '.join(missing)}}", file=sys.stderr)
        return 2
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_ref, path=str(target_dir), unzip=True, quiet=False)
    if not any(target_dir.iterdir()):
        print("Download completed but data/ is empty", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
        return {
            "status": "ready",
            "code": code,
            "downloaded_dataset_description": candidate.description,
            "evidence": [
                {
                    "source_type": "provider_shortcut",
                    "title": "Kaggle dataset URL from paper description",
                    "url": url,
                    "note": "Used deterministic Kaggle downloader because the description already contained a concrete Kaggle dataset URL.",
                }
            ],
            "reason": "Found an explicit Kaggle dataset URL in the dataset description; using the Kaggle API is the safest direct path.",
        }

    fred_csv_match = re.search(r"https?://fred\.stlouisfed\.org/graph/fredgraph\.csv\?id=[A-Za-z0-9._-]+", text)
    if fred_csv_match:
        url = fred_csv_match.group(0)
        code = _direct_download_script(url, required_env=[], provider="FRED static CSV")
        return {
            "status": "ready",
            "code": code,
            "downloaded_dataset_description": candidate.description,
            "evidence": [
                {
                    "source_type": "provider_shortcut",
                    "title": "Direct FRED CSV URL from paper description",
                    "url": url,
                    "note": "Used deterministic direct-download script because the description already contained a stable FRED CSV URL.",
                }
            ],
            "reason": "Found a stable direct FRED CSV URL in the dataset description.",
        }

    github_match = re.search(r"https?://github\.com/[^\s)]+(?:/releases/download/[^\s)]+|/raw/[^\s)]+|/archive/[^\s)]+)", text)
    if github_match:
        url = github_match.group(0)
        code = _direct_download_script(url, required_env=[], provider="GitHub direct artifact")
        return {
            "status": "ready",
            "code": code,
            "downloaded_dataset_description": candidate.description,
            "evidence": [
                {
                    "source_type": "provider_shortcut",
                    "title": "Direct GitHub artifact URL from paper description",
                    "url": url,
                    "note": "Used deterministic direct-download script because the description already contained a direct GitHub artifact URL.",
                }
            ],
            "reason": "Found a direct GitHub artifact URL in the dataset description.",
        }

    hf_match = re.search(r"https?://huggingface\.co/.+/resolve/.+", text)
    if hf_match:
        url = hf_match.group(0)
        code = _direct_download_script(url, required_env=["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"], provider="Hugging Face direct file")
        return {
            "status": "ready",
            "code": code,
            "downloaded_dataset_description": candidate.description,
            "evidence": [
                {
                    "source_type": "provider_shortcut",
                    "title": "Direct Hugging Face file URL from paper description",
                    "url": url,
                    "note": "Used deterministic direct-download script because the description already contained a direct Hugging Face file URL.",
                }
            ],
            "reason": "Found a direct Hugging Face file URL in the dataset description.",
        }

    return None


def _direct_download_script(url: str, *, required_env: list[str], provider: str) -> str:
    required_env_repr = repr(required_env)
    return f"""import os
import sys
from pathlib import Path

import requests


def main() -> int:
    target_dir = Path("data")
    target_dir.mkdir(parents=True, exist_ok=True)
    url = {url!r}
    required_env = {required_env_repr}
    if os.getenv("DRY_RUN") == "1":
        print("provider={provider}")
        print(f"url={{url}}")
        print(f"output_dir={{target_dir.resolve()}}")
        if required_env:
            print(f"optional_env={{','.join(required_env)}}")
        return 0
    headers = {{"User-Agent": "OpenFinAIGymDatasetDownloader/1.0"}}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {{token}}"
    response = requests.get(url, headers=headers, timeout=60, stream=True)
    if response.status_code >= 400:
        print(f"HTTP {{response.status_code}} for {{url}}", file=sys.stderr)
        return 2
    filename = Path(url.split("?", 1)[0]).name or "downloaded_artifact"
    output_path = target_dir / filename
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                handle.write(chunk)
    if output_path.stat().st_size <= 0:
        print("Downloaded file is empty", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


# Dataset labeler (post-download)


_ROLE_SET: frozenset[str] = frozenset(ROLE_VOCAB)

# Cap on auto-emitted auxiliary artifact entries. Wide-tree datasets
# (e.g. CIKM20 MAEC ~9k files) otherwise grow the manifest past 2 MB and
# blow Phase-4 prompt token budgets. Overflow is rolled up as a single
# ``labeler_notes`` line so the loader prompt still sees pattern+examples.
_MAX_AUTO_AUX_ENTRIES = 200


def label_dataset_artifacts(
    *,
    name: str,
    description: str,
    downloaded_dataset_description: str,
    interaction_model_hint: str,
    payload_files: list[str],
    preview: dict[str, Any],
    dataset_kind: str,
    llm: LLMService | None,
) -> dict[str, Any]:
    """Classify every landed file by role via a post-download LLM call.

    Returns a dict with keys ``manifest_status``, ``artifacts``
    (list of ``{role, path, format, shape?, columns?, dtype?,
    description?}``), ``labeler_notes``, plus a derived
    ``has_ground_truth`` (True iff at least one artifact has
    role=``ground_truth`` or role=``reference``), plus provider/model
    metadata the LLMService attaches automatically. The LLM no longer
    authors ``has_ground_truth`` — the role tags are the single
    source of truth and the bool is computed deterministically from
    them post-parsing.

    On any failure (no LLM, schema rejection, malformed output) returns a
    safe fallback payload with ``manifest_status="unresolved"`` so Phase 4
    hard-fails at assembly time rather than silently proceeding.
    """
    fallback: dict[str, Any] = {
        "manifest_status": "unresolved",
        "has_ground_truth": False,
        "artifacts": [],
        "labeler_notes": "llm_unavailable" if llm is None else "fallback",
    }
    normalized_files = [str(p).strip().replace("\\", "/") for p in payload_files or []]
    if llm is None or not normalized_files:
        return _normalize_labeler_payload(dict(fallback), normalized_files)
    payload = llm.complete_or_fallback(
        build_label_dataset_prompt(
            name=name,
            description=description,
            downloaded_dataset_description=downloaded_dataset_description,
            interaction_model_hint=interaction_model_hint,
            payload_files=normalized_files,
            preview=preview or {},
            dataset_kind=dataset_kind,
        ),
        dataset_label_schema(),
        fallback=dict(fallback),
    )
    return _normalize_labeler_payload(payload, normalized_files)


def _normalize_labeler_payload(
    payload: Any,
    payload_files: list[str],
) -> dict[str, Any]:
    """Defensive post-processing of the labeler's raw output.

    - Drops artifact entries whose ``path`` is not in ``payload_files``
      (rejects LLM hallucinations).
    - Coerces paths to POSIX and strips any leading ``data/`` prefix.
    - Canonicalizes ``format`` aliases against ``FORMAT_VOCAB``.
    - Auto-assigns ``role="auxiliary"`` for any real file the labeler
      didn't classify. Capped at ``_MAX_AUTO_AUX_ENTRIES`` full entries
      to keep wide-tree datasets from bloating the manifest; overflow
      paths land in a single ``labeler_notes`` summary line (count,
      format mix, sample paths) so the loader prompt can still see the
      pattern. LLM-classified entries — including LLM-explicit
      ``auxiliary`` ones — are never capped and keep their full
      ``shape``/``columns``/``description`` payload.
    - Guards against non-dict payloads.
    """
    if not isinstance(payload, dict):
        return {
            "manifest_status": "unresolved",
            "has_ground_truth": False,
            "artifacts": [],
            "labeler_notes": "labeler_returned_non_dict",
        }
    file_set = {_posix_relpath(p) for p in payload_files}
    raw_artifacts = payload.get("artifacts") or []
    if not isinstance(raw_artifacts, list):
        raw_artifacts = []
    seen_paths: set[str] = set()
    normalized_artifacts: list[dict[str, Any]] = []
    notes: list[str] = []
    raw_notes = str(payload.get("labeler_notes", "") or "").strip()
    if raw_notes:
        notes.append(raw_notes)
    for item in raw_artifacts:
        if not isinstance(item, dict):
            continue
        raw_path = _posix_relpath(str(item.get("path", "")))
        if not raw_path:
            continue
        if raw_path not in file_set:
            notes.append(f"dropped_hallucinated_path:{raw_path}")
            continue
        if raw_path in seen_paths:
            notes.append(f"duplicate_path:{raw_path}")
            continue
        role = str(item.get("role", "")).strip().lower()
        if role not in _ROLE_SET:
            notes.append(f"unknown_role:{role}->auxiliary:{raw_path}")
            role = "auxiliary"
        fmt = canonicalize_format(str(item.get("format", "")))
        if fmt == "other":
            inferred = infer_format_from_suffix(raw_path)
            if inferred != "other":
                fmt = inferred
        entry: dict[str, Any] = {"role": role, "path": raw_path, "format": fmt}
        shape = item.get("shape")
        if isinstance(shape, list):
            try:
                entry["shape"] = [int(v) for v in shape]
            except (TypeError, ValueError):
                pass
        columns = item.get("columns")
        if isinstance(columns, list):
            entry["columns"] = [str(v) for v in columns]
        dtype = item.get("dtype")
        if dtype not in (None, ""):
            entry["dtype"] = str(dtype)
        description = str(item.get("description", "")).strip()
        if description:
            entry["description"] = description
        normalized_artifacts.append(entry)
        seen_paths.add(raw_path)
    # Auto-classify unclassified real files as auxiliary so the manifest
    # covers what landed; cap at _MAX_AUTO_AUX_ENTRIES with rollup note for
    # wide-tree datasets (CIKM20 MAEC ~9k files) that would otherwise blow
    # Phase-4 prompt budgets.
    auto_aux_paths: list[str] = []
    for path in payload_files:
        rel = _posix_relpath(path)
        if rel and rel not in seen_paths:
            auto_aux_paths.append(rel)
            seen_paths.add(rel)
    kept = auto_aux_paths[:_MAX_AUTO_AUX_ENTRIES]
    overflow = auto_aux_paths[_MAX_AUTO_AUX_ENTRIES:]
    for rel in kept:
        normalized_artifacts.append(
            {
                "role": "auxiliary",
                "path": rel,
                "format": infer_format_from_suffix(rel),
            }
        )
    if auto_aux_paths:
        summary = f"auto_classified_auxiliary: count={len(auto_aux_paths)}"
        if overflow:
            fmt_counts = Counter(infer_format_from_suffix(p) for p in overflow)
            # Stratified sample so the LLM sees variety, not the alphabetical head.
            stride = max(1, len(overflow) // 5)
            examples = overflow[::stride][:5]
            summary += (
                f"; kept_full={len(kept)}"
                f"; truncated={len(overflow)}"
                f"; truncated_format_counts={dict(sorted(fmt_counts.items()))}"
                f"; truncated_examples={examples}"
            )
        notes.append(summary)
    manifest_status = str(payload.get("manifest_status", "")).strip().lower()
    if manifest_status not in {"labeled", "unresolved"}:
        manifest_status = "unresolved"
    # ``has_ground_truth`` is a derived mirror of role tags (single source
    # of truth): True iff any artifact has role=ground_truth or =reference.
    role_set = {a["role"] for a in normalized_artifacts}
    has_ground_truth = ("ground_truth" in role_set) or ("reference" in role_set)
    result: dict[str, Any] = {
        "manifest_status": manifest_status,
        "has_ground_truth": has_ground_truth,
        "artifacts": normalized_artifacts,
        "labeler_notes": "; ".join(notes),
    }
    provider = str(payload.get("provider", "")).strip()
    model = str(payload.get("model", "")).strip()
    if provider:
        result["provider"] = provider
    if model:
        result["model"] = model
    return result


def _posix_relpath(path: str) -> str:
    """Normalize a path to POSIX-style, relative (strip leading ``data/``)."""
    text = str(path or "").strip().replace("\\", "/").lstrip("/")
    if text.startswith("data/"):
        text = text[len("data/"):]
    return text
