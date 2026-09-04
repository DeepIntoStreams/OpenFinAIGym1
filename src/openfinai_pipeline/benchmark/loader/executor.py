"""Generate, validate, and stage per-task data loaders.

Installation executes the approved loader to materialize agent and verifier
HDF5 artifacts, then installs a deterministic runtime loader.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openfinai_pipeline.benchmark.loader.codegen import generate_loader_code
from openfinai_pipeline.benchmark.loader.reviewer import review_loader_code
from openfinai_pipeline.utils.logging import log_stage, truncate_oneline

if TYPE_CHECKING:  # avoid a circular import; types are only used for hints
    from openfinai_pipeline.benchmark.builders.task_builder import (
        DatasetArtifact,
        TaskCandidate,
    )

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LOADER_TEST_FILE = PROJECT_ROOT / "tests" / "test_loader_contract.py"
PHASE_TAG = "[phase4:benchmark]"


@dataclass
class LoaderResult:
    """Loader-generation outcome.

    Status is ready, unresolved, or not_required. Ready results carry the
    declared split policy and ground-truth provenance; installation verifies
    the policy against the loader's actual output.
    """

    status: str
    path: str = ""
    notes: str = ""
    provider: str = ""
    model: str = ""
    rounds_log: list[int] = field(default_factory=list)
    split_policy: str = ""
    ground_truth_provenance: dict[str, Any] | None = None


def run_loader_script(
    load_path: Path,
    data_dir: Path,
    *,
    timeout_sec: int = 120,
) -> str:
    """Compile and contract-test a loader, returning its subprocess log."""
    if not load_path.exists() or not load_path.read_text(encoding="utf-8").strip():
        return f"error: empty {load_path.name}"
    # Report syntax errors before starting pytest.
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", load_path.name],
            cwd=str(load_path.parent),
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return f"error: {exc}"
    if proc.returncode != 0:
        return (
            f"returncode={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    if not _LOADER_TEST_FILE.exists():
        return f"error: loader contract test missing at {_LOADER_TEST_FILE}"

    if not data_dir.exists():
        return f"error: dataset data dir missing at {data_dir}"

    env = os.environ.copy()
    extra_paths = os.pathsep.join([str(PROJECT_ROOT), str(PROJECT_ROOT / "src")])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{extra_paths}{os.pathsep}{existing}" if existing else extra_paths
    )

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(_LOADER_TEST_FILE),
                "-k",
                "TestLoaderContract",
                "--loader-module",
                str(load_path.resolve()),
                "--loader-data-dir",
                str(data_dir.resolve()),
                "-v",
                "--tb=short",
                "--no-header",
            ],
            cwd=str(PROJECT_ROOT),
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            env=env,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return f"error: pytest {exc}"

    return (
        f"returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _feedback_from_review(review: dict[str, Any]) -> str:
    """Compress a reviewer payload into prose the next round can consume."""
    analysis = str(review.get("analysis", "")).strip()
    issues = [str(x) for x in review.get("issues", []) if str(x).strip()]
    suggestions = [str(x) for x in review.get("suggestions", []) if str(x).strip()]
    parts: list[str] = []
    if analysis:
        parts.append("Reviewer analysis:")
        parts.append(analysis)
    if issues:
        parts.append("Issues:")
        parts.extend(f"- {x}" for x in issues)
    if suggestions:
        parts.append("Suggestions:")
        parts.extend(f"- {x}" for x in suggestions)
    return "\n".join(parts).strip()


def _llm_model_label(llm: Any | None) -> str:
    """Best-effort ``provider:model`` label for telemetry."""
    if llm is None:
        return ""
    router = getattr(llm, "_router", None)
    if router is None:
        return ""
    provider = str(getattr(router, "_provider_name", "")).strip()
    cfg = getattr(router, "_cfg", None)
    providers = getattr(cfg, "providers", {}) if cfg is not None else {}
    provider_cfg = providers.get(provider) if isinstance(providers, dict) else None
    model = (
        str(getattr(provider_cfg, "model", "")).strip()
        if provider_cfg is not None
        else ""
    )
    if provider and model:
        return f"{provider}:{model}"
    return model or provider


def _extract_provider_and_model(llm: Any | None) -> tuple[str, str]:
    label = _llm_model_label(llm)
    if not label:
        return "", ""
    if ":" in label:
        provider, model = label.split(":", 1)
        return provider.strip(), model.strip()
    return "", label


def _resolve_dataset_data_dir(dataset: "DatasetArtifact") -> Path | None:
    """Find the ``data`` ancestor of the dataset's first source path."""
    if not dataset.source_paths:
        return None
    try:
        first_path = Path(dataset.source_paths[0]).resolve()
    except Exception:  # noqa: BLE001 — best-effort
        return None
    for parent in [first_path, *first_path.parents]:
        if parent.name == "data" and parent.is_dir():
            return parent
    return None


def _read_labeler_notes(data_dir: Path) -> str:
    """Read labeler notes from the dataset manifest, if available."""
    manifest_path = data_dir.parent / "manifest.json"
    if not manifest_path.exists():
        return ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(manifest, dict):
        return ""
    return str(manifest.get("labeler_notes", "") or "").strip()


def _rewards_for_prompt(rewards: list[Any] | None) -> list[dict[str, Any]]:
    """Keep named reward dictionaries needed by the prompt."""
    if not rewards:
        return []
    out: list[dict[str, Any]] = []
    for entry in rewards:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        out.append(
            {
                "name": name,
                "description": str(entry.get("description", "")).strip(),
            }
        )
    return out


def _rewrite_loader_rounds(
    *,
    candidate: "TaskCandidate",
    dataset: "DatasetArtifact",
    data_dir: Path,
    load_path: Path,
    rounds_dir: Path,
    llm: Any | None,
    paper_context: str,
    labeler_notes: str,
    generation_rounds: int,
    interaction_model: str = "",
) -> list[dict[str, Any]]:
    """Run iterative loader generation, contract tests, and review.

    Numbered candidates remain in ``rounds_dir``; ``load_path`` always holds
    the latest candidate. A task-level interaction model takes precedence.
    """
    if generation_rounds < 1:
        raise ValueError("generation_rounds must be >= 1")
    rounds: list[dict[str, Any]] = []
    prev_script = ""
    prev_execution_log = ""
    prev_review = ""

    name = (dataset.name or "").strip()
    description = (dataset.downloaded_dataset_description or "").strip()
    downloaded_desc = description
    effective_interaction = (
        (interaction_model or "").strip()
        or (dataset.interaction_model or "").strip()
    )
    artifacts = list(dataset.artifacts or [])
    preview = dataset.preview

    rewards = _rewards_for_prompt(getattr(candidate, "rewards", []) or [])
    task_title = str(getattr(candidate, "title", "") or "").strip()
    ml_task_summary = str(getattr(candidate, "ml_task_summary", "") or "").strip()
    experiments = str(getattr(candidate, "experiments", "") or "").strip()

    for i in range(1, generation_rounds + 1):
        script_name = f"load{i}.py"
        payload = generate_loader_code(
            name=name,
            description=description,
            downloaded_dataset_description=downloaded_desc,
            interaction_model=effective_interaction,
            artifacts=artifacts,
            preview=preview,
            labeler_notes=labeler_notes,
            llm=llm,
            task_title=task_title,
            ml_task_summary=ml_task_summary,
            experiments=experiments,
            rewards=rewards,
            paper_context=paper_context,
            execution_log=prev_execution_log,
            previous_review=prev_review,
            previous_script=prev_script,
        )
        code = str(payload.get("code", ""))
        provenance = (
            payload.get("ground_truth_provenance") or {}
        ) if isinstance(payload.get("ground_truth_provenance"), dict) else {}

        # Retain the round and expose its candidate to the test harness.
        (rounds_dir / script_name).write_text(code, encoding="utf-8")
        load_path.write_text(code, encoding="utf-8")

        execution_log = (
            run_loader_script(load_path, data_dir)
            if code.strip()
            else "error: empty code"
        )
        review = review_loader_code(
            llm,
            loader_code=code,
            dataset_name=name,
            dataset_description=description,
            interaction_model=effective_interaction,
            artifacts=artifacts,
            labeler_notes=labeler_notes,
            derivation_rationale=str(payload.get("derivation_rationale", "")),
            ground_truth_provenance=provenance,
            task_title=task_title,
            ml_task_summary=ml_task_summary,
            experiments=experiments,
            rewards=rewards,
            paper_context=paper_context,
            execution_log=execution_log,
            previous_script=prev_script,
            previous_execution_log=prev_execution_log,
        )
        rounds.append(
            {
                "round": i,
                "script_name": script_name,
                "code": code,
                "execution_log": execution_log,
                "review": review,
                "spec": {
                    "derivation_rationale": str(
                        payload.get("derivation_rationale", "")
                    ),
                    "split_policy": str(payload.get("split_policy", "")),
                    "ground_truth_provenance": dict(provenance),
                },
            }
        )
        approved = bool(review.get("approved", False))
        if approved:
            log_stage(
                logger,
                "%s   loader round %d/%d -> approved",
                PHASE_TAG,
                i,
                generation_rounds,
            )
        else:
            log_stage(
                logger,
                "%s   loader round %d/%d -> rejected: %s",
                PHASE_TAG,
                i,
                generation_rounds,
                _short_reject_reason_loader(review, execution_log),
            )
        prev_script = code
        prev_execution_log = execution_log
        prev_review = _feedback_from_review(review)
        if approved:
            break
    return rounds


def _short_reject_reason_loader(
    review: dict[str, Any],
    execution_log: str,
) -> str:
    issues = review.get("issues") or []
    if issues:
        return truncate_oneline(str(issues[0]))
    log = (execution_log or "").strip()
    if log.startswith("error:"):
        return truncate_oneline(log.splitlines()[0])
    if "returncode=" in log and "returncode=0" not in log.split("\n", 1)[0]:
        for line in log.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(("returncode=", "stdout:", "stderr:")):
                return truncate_oneline("pytest: " + stripped)
        return "pytest contract failed"
    analysis = str(review.get("analysis", "")).strip()
    if analysis:
        return truncate_oneline(analysis)
    return "no specific issue surfaced"


# Contract tests import this ground-truth requirement through benchmark.loader.
LOADER_REQUIRES_GT_MODELS: frozenset[str] = frozenset({"forecasting", "generative"})


def generate_task_loader(
    *,
    candidate: "TaskCandidate",
    dataset: "DatasetArtifact",
    llm: Any | None,
    staging_dir: Path,
    paper_context: str = "",
    generation_rounds: int = 3,
    task_interaction_model: str = "",
) -> LoaderResult:
    """Generate and pytest-validate load.py in the staging directory.

    Forecasting and generative tasks require a loader regardless of the source
    artifact layout. Other interaction models return not_required. Installation
    executes ready loaders and replaces them with the runtime loader.
    """
    provider, model = _extract_provider_and_model(llm)
    effective_interaction = (
        (task_interaction_model or "").strip().lower()
        or (dataset.interaction_model or "").strip().lower()
    )

    staged_load = staging_dir / "load.py"

    # TODO(multi-dataset): support one loader per dataset when tasks need it.

    if effective_interaction not in LOADER_REQUIRES_GT_MODELS:
        # Do not ship a stale loader for a model that does not use one.
        if staged_load.exists():
            try:
                staged_load.unlink()
            except OSError:
                pass
        notes = (
            f"interaction_model={effective_interaction!r} does not require "
            "a per-task load.py (trading/gym tasks stream data through the "
            "gym loop; only forecasting/generative tasks need a loader)."
        )
        return LoaderResult(
            status="not_required",
            path="",
            notes=notes,
            provider="",
            model="",
            rounds_log=[],
        )

    data_dir = _resolve_dataset_data_dir(dataset)
    if data_dir is None:
        return LoaderResult(
            status="unresolved",
            path="",
            notes=(
                "could not resolve dataset data/ subdir from "
                "dataset.source_paths — Phase 2 may not have published this dataset"
            ),
            provider=provider,
            model=model,
            rounds_log=[],
        )

    staging_dir.mkdir(parents=True, exist_ok=True)
    rounds_dir = staging_dir / ".loader_rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)

    labeler_notes = _read_labeler_notes(data_dir)

    rounds = _rewrite_loader_rounds(
        candidate=candidate,
        dataset=dataset,
        data_dir=data_dir,
        load_path=staged_load,
        rounds_dir=rounds_dir,
        llm=llm,
        paper_context=paper_context,
        labeler_notes=labeler_notes,
        generation_rounds=generation_rounds,
        interaction_model=effective_interaction,
    )

    final = rounds[-1] if rounds else {}
    approved = bool(final.get("review", {}).get("approved", False))
    final_spec = dict(final.get("spec") or {})
    final_log = str(final.get("execution_log", ""))

    # Persist the review trace beside the loader.
    review_log_path = rounds_dir / "loader_review.json"
    review_log_path.write_text(
        json.dumps(
            {
                "rounds": [
                    {
                        "round": int(r["round"]),
                        "script_name": str(r.get("script_name", "")),
                        "approved": bool(r["review"].get("approved", False)),
                        "analysis": str(r["review"].get("analysis", "")),
                        "issues": list(r["review"].get("issues", [])),
                        "suggestions": list(r["review"].get("suggestions", [])),
                        "execution_log_tail": (
                            str(r.get("execution_log", ""))[-800:]
                            if r.get("execution_log")
                            else ""
                        ),
                        "spec": dict(r.get("spec") or {}),
                    }
                    for r in rounds
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if approved:
        approved_provenance = final_spec.get("ground_truth_provenance")
        if not isinstance(approved_provenance, dict):
            approved_provenance = None
        return LoaderResult(
            status="ready",
            path=str(staged_load.resolve()),
            notes=str(final_spec.get("derivation_rationale", "")).strip(),
            provider=provider,
            model=model,
            rounds_log=[int(r["round"]) for r in rounds],
            split_policy=str(final_spec.get("split_policy", "")).strip(),
            ground_truth_provenance=(
                dict(approved_provenance) if approved_provenance else None
            ),
        )

    last_log_tail = final_log[-400:] if final_log else ""
    return LoaderResult(
        status="unresolved",
        path=str(staged_load.resolve()) if staged_load.exists() else "",
        notes=(
            "loader generation did not pass the contract after "
            f"{len(rounds)} round(s). Tail of last execution log: "
            f"{last_log_tail!r}"
        ),
        provider=provider,
        model=model,
        rounds_log=[int(r["round"]) for r in rounds],
    )
