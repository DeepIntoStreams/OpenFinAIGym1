import json
import logging
import os
import re
import site
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from openfinai_pipeline.corpus.aggregator import (
    DatasetCandidate,
    RewardCandidate,
)
from openfinai_pipeline.corpus.codegen import (
    generate_download_code,
    generate_reward_code,
    label_dataset_artifacts,
)
from openfinai_pipeline.corpus.contract_validator import (
    validate_reward_contract,
)
from openfinai_pipeline.corpus.reviewer import (
    review_download_script,
    review_reward_code,
)
from openfinai_pipeline.utils.logging import log_detail, log_stage, truncate_oneline
from openfinai_pipeline.utils.sandbox import Sandbox

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"
_REWARD_TEST_FILE = PROJECT_ROOT / "tests" / "test_reward_bank.py"
_LOG_SEPARATOR = "=" * 78
PHASE2_TAG = "[phase2:datasets]"
PHASE3_TAG = "[phase3:rewards]"


def _llm_model_label(llm: Any | None) -> str:
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


def _format_log_block(title: str, payload: dict[str, Any]) -> str:
    lines = [_LOG_SEPARATOR, title]
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            if value:
                for item in value:
                    lines.append(f"- {item}")
            else:
                lines.append("(empty list)")
        else:
            rendered = str(value).strip() if value is not None else ""
            lines.append(f"{key}: {rendered or '(empty)'}")
    lines.append(_LOG_SEPARATOR)
    return "\n".join(lines)


def _env_with_project_dotenv(env: dict[str, str]) -> dict[str, str]:
    merged = dict(env)
    if not PROJECT_ENV_FILE.exists():
        return merged
    for raw_line in PROJECT_ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in merged:
            continue
        merged[key] = value
    return merged


def run_download_script(
    dataset_dir: Path,
    script_name: str,
    timeout_sec: int,
    dry_run: bool,
    *,
    include_runtime_details: bool = True,
) -> str:
    script_path = dataset_dir / script_name
    if not script_path.exists() or not script_path.read_text(encoding="utf-8").strip():
        return f"error: empty {script_name}"
    env = _env_with_project_dotenv(os.environ.copy())
    env["DRY_RUN"] = "1" if dry_run else "0"
    runtime_details = _python_runtime_details()
    try:
        proc = subprocess.run(
            [sys.executable, script_name],
            cwd=str(dataset_dir),
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired as e:  # pragma: no cover
        # Emit a deterministic 'timeout:' marker so _classify_download_failures
        # tags this round as 'timeout' regardless of how Python phrases the
        # underlying exception ('timed out after N seconds' vs 'timeout').
        return (
            f"error: timeout: subprocess exceeded {timeout_sec}s wall clock; {e}"
        )
    except (subprocess.SubprocessError, OSError) as e:  # pragma: no cover
        return f"error: {e}"
    prefix = f"{runtime_details}\n" if include_runtime_details else ""
    return f"{prefix}returncode={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def _python_runtime_details() -> str:
    payload = _python_runtime_payload()
    return (
        f"python_executable={payload['python_executable']}\n"
        f"python_version={payload['python_version']}\n"
        f"site_packages={payload['site_packages']}"
    )


def _python_runtime_payload() -> dict[str, str]:
    python_version = sys.version.splitlines()[0]
    try:
        site_packages = site.getsitepackages()
    except Exception:
        site_packages = []
    primary_site_packages = site_packages[0] if site_packages else ""
    return {
        "python_executable": sys.executable,
        "python_version": python_version,
        "site_packages": primary_site_packages,
    }


def _collect_payload_files(data_dir: Path) -> list[str]:
    return [
        path.relative_to(data_dir).as_posix()
        for path in sorted(data_dir.rglob("*"))
        if path.is_file()
    ]


def _is_archive_path(path: Path) -> bool:
    name = path.name.lower()
    return any(
        name.endswith(suffix)
        for suffix in (
            ".zip",
            ".tar",
            ".tar.gz",
            ".tgz",
            ".tar.bz2",
            ".tbz2",
            ".tar.xz",
            ".txz",
        )
    )


def _publish_dataset_payload(
    *,
    source_data_dir: Path,
    target_data_dir: Path,
) -> list[str]:
    target_data_dir.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source_data_dir.rglob("*")):
        if not source_path.is_file():
            continue
        rel_path = source_path.relative_to(source_data_dir)
        if _is_archive_path(source_path):
            extract_dir = target_data_dir / rel_path.parent
            extract_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.unpack_archive(str(source_path), str(extract_dir))
            except (shutil.ReadError, ValueError, OSError):
                logger.warning("Failed to unpack dataset archive %s", source_path)
            continue
        dest_path = target_data_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
    return _collect_payload_files(target_data_dir)


def _preview_text_file(path: Path, max_lines: int = 5, max_chars: int = 600) -> dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    lines = content.splitlines()[:max_lines]
    preview = "\n".join(lines)
    if len(preview) > max_chars:
        preview = preview[:max_chars].rstrip() + "..."
    return {"preview_text": preview}


# Caps for CSV preview. The Phase 4 loader uses these to commit to a target
# column and a feature-leakage policy without executing code, so we want
# enough signal to disambiguate target candidates and spot transforms of the
# target — but not so much that wide / long datasets blow up the prompt.
_CSV_PREVIEW_LINES = 10
_CSV_PREVIEW_CHARS = 1500
_CSV_DESCRIBE_COLS_CAP = 30
_CSV_NULL_COUNT_COLS_CAP = 60
_CSV_READ_ROWS_CAP = 200_000


def _preview_csv_file(path: Path) -> dict[str, Any]:
    """Rich CSV preview: head text + columns + dtypes + nulls + numeric describe.

    Augments the cheap text head with structured fields read via pandas:
    full column list, per-column dtypes, non-zero null counts, and a
    ``describe()`` summary for up to ``_CSV_DESCRIBE_COLS_CAP`` numeric
    columns. Falls back to text-only when pandas is unavailable or the
    file can't be parsed as CSV.

    Sampling: the pandas read is capped at ``_CSV_READ_ROWS_CAP`` rows so
    pathologically large CSVs don't dominate the preview pass. When the
    cap fires, ``row_count_truncated_at`` records it so the LLM knows
    the stats reflect a head sample. Stat values are rounded to 6 sig
    figs — full precision isn't useful for picking a target column.
    """
    head_text: str = ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()[:_CSV_PREVIEW_LINES]
        head_text = "\n".join(lines)
        if len(head_text) > _CSV_PREVIEW_CHARS:
            head_text = head_text[:_CSV_PREVIEW_CHARS].rstrip() + "..."
    except OSError:
        return {}

    out: dict[str, Any] = {"preview_text": head_text}
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        return out
    try:
        df = pd.read_csv(path, nrows=_CSV_READ_ROWS_CAP)
    except Exception:  # noqa: BLE001 — fall back to text on any parse error
        return out
    if df.shape[1] == 0:
        return out

    out["columns"] = [str(c) for c in df.columns]
    out["row_count_sampled"] = int(df.shape[0])
    if df.shape[0] >= _CSV_READ_ROWS_CAP:
        out["row_count_truncated_at"] = _CSV_READ_ROWS_CAP
    out["dtypes"] = {str(c): str(df[c].dtype) for c in df.columns}

    # Only emit non-zero null counts to keep the payload small on clean
    # datasets; cap the column count so a wide-and-sparse CSV doesn't
    # dominate the prompt.
    null_counts = df.isna().sum()
    nz = [(str(c), int(null_counts[c])) for c in df.columns if int(null_counts[c]) > 0]
    if nz:
        out["null_counts"] = dict(nz[:_CSV_NULL_COUNT_COLS_CAP])
        if len(nz) > _CSV_NULL_COUNT_COLS_CAP:
            out["null_counts_overflow"] = len(nz) - _CSV_NULL_COUNT_COLS_CAP

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        described_cols = list(numeric_df.columns[:_CSV_DESCRIBE_COLS_CAP])
        described = numeric_df[described_cols].describe()
        out["describe"] = {
            str(col): {
                str(stat): _finite_or_none(value)
                for stat, value in described[col].items()
            }
            for col in described.columns
        }
        if numeric_df.shape[1] > _CSV_DESCRIBE_COLS_CAP:
            out["describe_overflow"] = int(
                numeric_df.shape[1] - _CSV_DESCRIBE_COLS_CAP
            )
    return out


def _finite_or_none(value: Any) -> float | None:
    """JSON-safe describe scalar: emit None for NaN/±inf, else round to 6 sig figs.

    ``json.dumps`` emits NaN/Infinity as non-standard tokens that some
    downstream consumers reject; rounding also keeps the prompt readable
    (full float precision isn't useful for picking a target column).
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    import math

    if not math.isfinite(f):
        return None
    return float(f"{f:.6g}")


def _preview_array_file(path: Path) -> dict[str, Any]:
    try:
        import numpy as np

        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
            return {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
            }
        if path.suffix == ".npz":
            with np.load(path) as data:
                members = {
                    key: {
                        "dtype": str(data[key].dtype),
                        "shape": list(data[key].shape),
                    }
                    for key in data.files[:5]
                }
            return {"members": members}
    except Exception:
        return {}
    return {}


# Max HDF5 datasets enumerated per file in ``_preview_hdf5_file``.
_HDF5_DATASETS_CAP = 20


def _preview_parquet_file(path: Path) -> dict[str, Any]:
    """Preview a parquet file via its self-describing schema.

    Parquet always carries a typed column schema in the file footer, so
    this read is deterministic — no delimiter/encoding sniffing involved.
    Returns ``{}`` when ``pyarrow`` is unavailable or the file is malformed.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError:
        return {}
    try:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        out: dict[str, Any] = {
            "columns": [field.name for field in schema],
            "dtypes": {field.name: str(field.type) for field in schema},
            "row_groups": int(pf.num_row_groups),
        }
        if pf.metadata is not None:
            out["row_count"] = int(pf.metadata.num_rows)
        return out
    except Exception:
        return {}


def _preview_arrow_or_feather_file(path: Path) -> dict[str, Any]:
    """Preview an Arrow IPC file or Feather file via its schema.

    Modern pyarrow's ``feather.read_table`` reads both Feather v1 and
    Arrow IPC (Feather v2) transparently, so a single helper covers
    ``.feather``, ``.arrow``, and ``.ipc``. Falls back to ``{}`` when
    pyarrow is unavailable or the file can't be opened.
    """
    try:
        import pyarrow.feather as feather  # type: ignore[import-not-found]
    except ImportError:
        return {}
    try:
        table = feather.read_table(str(path), memory_map=True)
        schema = table.schema
        return {
            "columns": [field.name for field in schema],
            "dtypes": {field.name: str(field.type) for field in schema},
            "row_count": int(table.num_rows),
        }
    except Exception:
        return {}


def _preview_hdf5_file(path: Path) -> dict[str, Any]:
    """Preview an HDF5 file by enumerating its dataset hierarchy.

    HDF5 is hierarchical (groups containing datasets). We list every
    Dataset's ``name`` (HDF5 path), ``shape``, ``dtype``. Capped at
    ``_HDF5_DATASETS_CAP`` to keep wide HDF5 archives from blowing up
    the prompt; an ``overflow`` count is recorded when the cap fires.
    Optional dependency — returns ``{}`` if ``h5py`` is not installed.
    """
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError:
        return {}
    try:
        datasets: list[dict[str, Any]] = []
        overflow = 0

        def _visit(name: str, obj: Any) -> None:
            nonlocal overflow
            if not isinstance(obj, h5py.Dataset):
                return
            if len(datasets) >= _HDF5_DATASETS_CAP:
                overflow += 1
                return
            datasets.append(
                {
                    "name": name,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }
            )

        with h5py.File(path, "r") as fh:
            fh.visititems(_visit)
        out: dict[str, Any] = {"datasets": datasets}
        if overflow:
            out["datasets_overflow"] = overflow
        return out
    except Exception:
        return {}


def _build_payload_preview(data_dir: Path, payload_files: list[str]) -> dict[str, Any]:
    previews: list[dict[str, Any]] = []
    for relpath in payload_files[:5]:
        path = data_dir / relpath
        entry: dict[str, Any] = {
            "path": relpath,
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        suffix = path.suffix.lower()
        if suffix == ".csv":
            entry.update(_preview_csv_file(path))
        elif suffix in {".txt", ".json", ".jsonl", ".md"}:
            entry.update(_preview_text_file(path))
        elif suffix in {".npy", ".npz"}:
            entry.update(_preview_array_file(path))
        elif suffix in {".parquet", ".pq"}:
            entry.update(_preview_parquet_file(path))
        elif suffix in {".feather", ".arrow", ".ipc"}:
            entry.update(_preview_arrow_or_feather_file(path))
        elif suffix in {".h5", ".hdf5"}:
            entry.update(_preview_hdf5_file(path))
        previews.append(entry)
    return {"files": previews}


def _build_usage_manifest(name: str, task_dirs: list[str]) -> dict[str, Any]:
    used_by_papers = []
    seen: set[str] = set()
    for task_dir in task_dirs:
        clean = str(task_dir).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        used_by_papers.append({"paper_dir": clean})
    return {
        "name": str(name).strip(),
        "used_by_papers": used_by_papers,
    }


def _build_full_manifest(
    candidate: DatasetCandidate,
    *,
    dataset_dir: Path | None,
    data_dir: Path | None,
    payload_files: list[str],
    preview: dict[str, Any],
    downloaded_dataset_description: str,
    llm: Any | None,
) -> dict[str, Any]:
    """Produce the full dataset manifest (usage + semantic roles).

    Layers the usage manifest (``{name, used_by_papers}``) with the
    paper-summary description, the LLM's pre-execution prose, and the
    post-download labeler's role assignments. This is the authoritative
    manifest written to ``<dataset_dir>/manifest.json``.

    Per-task ``load.py`` generation is owned by Phase 4 — see
    :mod:`openfinai_pipeline.benchmark.loader`. Phase 2 stops at the
    labeled manifest.
    """
    del dataset_dir, data_dir  # retained for API stability with prior callers
    base = _build_usage_manifest(candidate.name, candidate.task_dirs)
    base["description"] = str(candidate.description or "").strip()
    base["downloaded_dataset_description"] = str(
        downloaded_dataset_description or ""
    ).strip()
    base["dataset_kind"] = str(candidate.dataset_kind or "").strip()
    base["interaction_model"] = str(candidate.task_family or "").strip()
    labeler_result = label_dataset_artifacts(
        name=candidate.name,
        description=candidate.description,
        downloaded_dataset_description=downloaded_dataset_description,
        interaction_model_hint=str(candidate.task_family or "").strip(),
        payload_files=list(payload_files or []),
        preview=preview or {},
        dataset_kind=str(candidate.dataset_kind or "").strip(),
        llm=llm,
    )
    base["has_ground_truth"] = bool(labeler_result.get("has_ground_truth"))
    base["manifest_status"] = str(
        labeler_result.get("manifest_status", "") or "unresolved"
    )
    base["artifacts"] = list(labeler_result.get("artifacts") or [])
    base["labeler_notes"] = str(labeler_result.get("labeler_notes", "") or "")
    base["labeler_provider"] = str(labeler_result.get("provider", "") or "")
    base["labeler_model"] = str(labeler_result.get("model", "") or "")
    return base


def run_reward_script(
    reward_dir: Path, script_name: str, *, rewards_root: Path | None = None
) -> str:
    """Compile-check and pytest-validate a reward_fn script.

    *rewards_root* is the directory containing ``reward_bank.py`` — it is
    added to PYTHONPATH so the reward_fn can ``from reward_bank import Loss``.
    When *None* it falls back to heuristic parent-dir detection.
    """
    script_path = reward_dir / script_name
    if not script_path.exists() or not script_path.read_text(encoding="utf-8").strip():
        return f"error: empty {script_name}"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", script_name],
            cwd=str(reward_dir),
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError) as e:
        return f"error: {e}"
    if proc.returncode != 0:
        return f"returncode={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    # Phase 2: pytest validation using the unified test contract
    repo_root = PROJECT_ROOT
    test_file = _REWARD_TEST_FILE
    if not test_file.exists():
        return f"error: reward contract test missing at {test_file}"
    reward_module_path = str((reward_dir / script_name).resolve())

    # Resolve the directory containing reward_bank.py for PYTHONPATH
    if rewards_root is not None:
        eval_root = rewards_root
    else:
        eval_root = (
            reward_dir.parent
            if reward_dir.name not in ("evaluation", "eval", "rewards")
            else reward_dir
        )
    env = os.environ.copy()
    extra_paths = os.pathsep.join([str(repo_root), str(eval_root)])
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
                str(test_file),
                "-k",
                "TestUnifiedRewardContract",
                "--reward-module",
                reward_module_path,
                "-v",
                "--tb=short",
                "--no-header",
            ],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            env=env,
        )
    except (subprocess.SubprocessError, OSError) as e:
        return f"error: pytest {e}"

    return f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def write_dataset_asset(
    datasets_root: Path,
    candidate: DatasetCandidate,
    *,
    llm: Any | None,
    timeout_sec: int,
    execute_download: bool,
    staging_root: Path | None = None,
    dataset_generation_rounds: int = 2,
) -> dict[str, Any]:
    """Generate and validate a dataset download script through configurable rounds.

    Intermediate round scripts and data are written to
    *staging_root*/datasets/<directory_name>/
    (defaults to ``datasets_root/../.staging/datasets/<directory_name>/``).
    If the final round is approved and successful the dataset is promoted
    (copied) to ``datasets_root/<directory_name>/``.

    ``dataset_generation_rounds`` budgets the download-script generate/fix loop.
    Per-task ``load.py`` generation lives in Phase 4 (see
    :mod:`openfinai_pipeline.benchmark.loader`); Phase 2 stops at the
    labeled manifest.
    """
    # resolve staging directory
    if staging_root is None:
        staging_root = datasets_root.parent / ".staging"
    staging_dir = staging_root / "datasets" / candidate.directory_name
    staging_dir.mkdir(parents=True, exist_ok=True)

    # run generation rounds in staging
    rounds = _rewrite_download_rounds(
        dataset_dir=staging_dir,
        candidate=candidate,
        llm=llm,
        timeout_sec=timeout_sec,
        execute_download=execute_download,
        dataset_generation_rounds=dataset_generation_rounds,
    )
    final = rounds[-1]
    final_success = bool(final.get("real_success", False))
    final_script = str(final.get("script_name", "download3.py"))
    final_data_dir_name = str(final.get("data_dir_name", ""))
    approved = bool(final.get("review", {}).get("approved", False))
    generation_status = str(final.get("generation_status", "")).strip()
    download_attempted = bool(
        execute_download and str(final.get("code", "")).strip()
    )
    if not generation_status:
        if final_success and approved:
            generation_status = "materialized"
        elif not approved:
            generation_status = "not_approved"
        else:
            generation_status = "failed_execution"
    elif final_success and approved:
        generation_status = "materialized"
    elif generation_status == "ready" and final_success and not approved:
        generation_status = "not_approved"
    elif generation_status == "ready" and not final_success:
        generation_status = "failed_execution"

    # write review + logs + manifest to staging
    _write_download_round_logs(staging_dir, rounds)
    (staging_dir / "download_review.json").write_text(
        json.dumps(
            {
                "rounds": [
                    {
                        "round": int(r["round"]),
                        "analysis": str(r["review"].get("analysis", "")),
                        "approved": bool(r["review"].get("approved", False)),
                        "issues": list(r["review"].get("issues", [])),
                        "suggestions": list(r["review"].get("suggestions", [])),
                        "provider": str(r["review"].get("provider", "")),
                        "model": str(r["review"].get("model", "")),
                    }
                    for r in rounds
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    staging_data_dir = staging_dir / final_data_dir_name if final_data_dir_name else None
    payload_files = (
        _collect_payload_files(staging_data_dir)
        if final_success and staging_data_dir and staging_data_dir.exists()
        else []
    )
    # Staging manifest is usage-only; full labeling runs once below
    # against the published data dir on success+approval, so we don't
    # burn the labeler call on potentially-rejected scripts.
    thin_manifest = _build_usage_manifest(candidate.name, candidate.task_dirs)
    (staging_dir / "manifest.json").write_text(
        json.dumps(thin_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # publish approved + successful dataset to final location
    published_payload_files: list[str] = []
    publish_script_path = ""
    materialized = False
    if final_success and approved and staging_data_dir is not None and staging_data_dir.exists():
        final_dir = datasets_root / candidate.directory_name
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        published_payload_files = _publish_dataset_payload(
            source_data_dir=staging_data_dir,
            target_data_dir=final_dir / "data",
        )
        src_script = staging_dir / final_script
        if src_script.exists():
            dst_script = final_dir / "download.py"
            shutil.copy2(src_script, dst_script)
            publish_script_path = str(dst_script.resolve())
        # Label the published payload (not staging) so the labeler sees
        # the final on-disk layout.
        published_data_dir = final_dir / "data"
        published_preview = (
            _build_payload_preview(published_data_dir, published_payload_files)
            if published_payload_files and published_data_dir.exists()
            else {"files": []}
        )
        full_manifest = _build_full_manifest(
            candidate,
            dataset_dir=final_dir,
            data_dir=published_data_dir,
            payload_files=published_payload_files,
            preview=published_preview,
            downloaded_dataset_description=str(
                final.get("downloaded_dataset_description", "")
            ).strip(),
            llm=llm,
        )
        (final_dir / "manifest.json").write_text(
            json.dumps(full_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        materialized = bool(published_payload_files) and bool(publish_script_path)
        if not materialized:
            shutil.rmtree(final_dir, ignore_errors=True)
        else:
            logger.info(
                "Promoted approved dataset %s -> %s (manifest_status=%s has_gt=%s)",
                staging_dir,
                final_dir,
                full_manifest.get("manifest_status"),
                full_manifest.get("has_ground_truth"),
            )

    final_dir = datasets_root / candidate.directory_name if materialized else staging_dir
    final_data_dir = final_dir / "data" if materialized else staging_data_dir
    final_payload_files = published_payload_files if materialized else payload_files
    dataset_paths = (
        [
            str((final_data_dir / rel_path).resolve())
            for rel_path in final_payload_files
            if final_data_dir is not None and (final_data_dir / rel_path).exists()
        ]
        if final_payload_files
        else []
    )
    preview = (
        _build_payload_preview(final_data_dir, final_payload_files)
        if final_payload_files and final_data_dir is not None and final_data_dir.exists()
        else {"files": []}
    )
    download_script_path = publish_script_path if materialized else ""
    if final_success and approved and not materialized:
        generation_status = "failed_execution"

    return {
        "final_success": final_success,
        "final_script": "download.py" if materialized else final_script,
        "approved": approved,
        "generation_status": generation_status,
        "payload_files": final_payload_files,
        "preview": preview.get("files", []) if isinstance(preview, dict) else [],
        "downloaded_dataset_description": str(
            final.get("downloaded_dataset_description", "")
        ).strip(),
        "download_attempted": download_attempted,
        "download_script_path": download_script_path,
        "dataset_paths": dataset_paths,
        "final_review": final.get("review", {}),
        "model": _llm_model_label(llm),
        "staging_dir": str(staging_dir),
    }


def write_reward_asset(
    rewards_root: Path,
    reward: RewardCandidate,
    llm: Any | None,
    staging_root: Path | None = None,
    reward_generation_rounds: int = 3,
    generated_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate and validate a reward_fn through configurable rounds.

    Intermediate round scripts are written to *staging_root*/rewards/<slug>/
    (defaults to ``rewards_root/../.staging/rewards/<slug>/``).
    If the final round is approved the reward_fn is promoted to
    *generated_dir*/<slug>.py (defaults to ``rewards_root/generated/<slug>.py``).
    """
    import shutil

    from openfinai_pipeline.corpus.catalog import slugify

    reward_slug = slugify(reward.name)

    # resolve staging directory
    if staging_root is None:
        staging_root = rewards_root.parent / ".staging"
    staging_dir = staging_root / "rewards" / reward_slug
    staging_dir.mkdir(parents=True, exist_ok=True)

    # run generation rounds in staging
    rounds = _rewrite_reward_rounds(
        reward_dir=staging_dir,
        reward=reward,
        llm=llm,
        rewards_root=rewards_root,
        reward_generation_rounds=reward_generation_rounds,
    )
    final = rounds[-1]
    approved = bool(final.get("review", {}).get("approved", False))
    final_script = str(final.get("script_name", f"reward_fn{reward_generation_rounds}.py"))

    # write review + manifest to staging
    (staging_dir / "reward_review.json").write_text(
        json.dumps(
            {
                "rounds": [
                    {
                        "round": int(r["round"]),
                        "script_name": str(r.get("script_name", "")),
                        "analysis": str(r["review"].get("analysis", "")),
                        "approved": bool(r["review"].get("approved", False)),
                        "issues": list(r["review"].get("issues", [])),
                        "suggestions": list(r["review"].get("suggestions", [])),
                        "provider": str(r["review"].get("provider", "")),
                        "model": str(r["review"].get("model", "")),
                    }
                    for r in rounds
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (staging_dir / "manifest.json").write_text(
        json.dumps(
            {
                "name": reward.name,
                "slug": reward_slug,
                "aliases": reward.aliases,
                "description": reward.description,
                "approved": approved,
                "reward_fn": f"{reward_slug}.py" if approved else "",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # promote approved reward to generated/ subdirectory
    reward_path = ""
    if approved:
        promote_dir = generated_dir if generated_dir is not None else (rewards_root / "generated")
        promote_dir.mkdir(parents=True, exist_ok=True)
        src = staging_dir / final_script
        dst = promote_dir / f"{reward_slug}.py"
        shutil.copy2(str(src), str(dst))
        reward_path = str(dst.resolve())
        logger.info(
            "Promoted approved reward %s -> %s",
            src,
            dst,
        )
        # Clean up any stale per-reward ``<slug>.spec.json``; the unified
        # spec now lives in ``generated_rewards.json``.
        legacy_spec = promote_dir / f"{reward_slug}.spec.json"
        if legacy_spec.exists():
            try:
                legacy_spec.unlink()
            except OSError:
                logger.warning(
                    "Could not remove stale spec file %s",
                    legacy_spec,
                )

    result: dict[str, Any] = {
        "approved": approved,
        "slug": reward_slug,
        "reward_fn": f"{reward_slug}.py" if approved else final_script,
        "script_path": reward_path,
        "staging_dir": str(staging_dir),
    }
    if approved:
        result["spec"] = {
            "class_name": _first_loss_class_name(final.get("code", "")),
            "module_name": reward_slug,
            "base_class": "Loss",
            "default_params": _infer_default_params_from_source(
                final.get("code", "")
            ),
        }
    return result


def _first_loss_class_name(source_code: str) -> str:
    """AST-parse ``source_code`` and return the first ``Loss`` subclass name.

    Empty string if none found. Used to populate the ``spec.class_name``
    without needing to import the generated module.
    """
    if not source_code:
        return ""
    import ast
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if base_name == "Loss":
                return node.name
    return ""


def _infer_default_params_from_source(source_code: str) -> dict[str, Any]:
    """Extract literal defaults from the first Loss subclass's ``__init__``.

    Used at codegen-time to snapshot the LLM's chosen hyperparameter
    defaults into the catalog entry, where ``task_builder`` can later
    layer per-task overrides on top. Skips ``self`` and the ``name``
    Loss-base identifier kwarg. Skips non-JSON-safe values (callables,
    tensors, etc.) since the catalog must round-trip through JSON.
    """
    if not source_code:
        return {}
    import ast
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return {}
    out: dict[str, Any] = {}
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        is_loss = False
        for base in cls.bases:
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if base_name == "Loss":
                is_loss = True
                break
        if not is_loss:
            continue
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if fn.name != "__init__":
                continue
            pos_args = list(fn.args.posonlyargs) + list(fn.args.args)
            n_pos = len(pos_args)
            n_defaults = len(fn.args.defaults)
            pos_defaults: list[Any] = (
                [None] * (n_pos - n_defaults) + list(fn.args.defaults)
            )
            for arg, default_node in zip(pos_args, pos_defaults):
                if arg.arg == "self" or arg.arg == "name":
                    continue
                if default_node is None:
                    continue
                try:
                    value = ast.literal_eval(default_node)
                except (ValueError, SyntaxError, TypeError):
                    continue
                if value is None:
                    continue
                if not _is_json_safe_default(value):
                    continue
                out[arg.arg] = _normalize_default_for_json(value)
            for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
                if arg.arg == "self" or arg.arg == "name":
                    continue
                if default is None:
                    continue
                try:
                    value = ast.literal_eval(default)
                except (ValueError, SyntaxError, TypeError):
                    continue
                if value is None:
                    continue
                if not _is_json_safe_default(value):
                    continue
                out[arg.arg] = _normalize_default_for_json(value)
            return out
    return out


def _is_json_safe_default(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe_default(v) for v in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _is_json_safe_default(v) for k, v in value.items()
        )
    return False


def _normalize_default_for_json(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_normalize_default_for_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_default_for_json(v) for k, v in value.items()}
    return value


def _rewrite_download_rounds(
    *,
    dataset_dir: Path,
    candidate: DatasetCandidate,
    llm: Any | None,
    timeout_sec: int,
    execute_download: bool,
    dataset_generation_rounds: int,
) -> list[dict[str, Any]]:
    if dataset_generation_rounds < 1:
        raise ValueError("dataset_generation_rounds must be >= 1")
    rounds: list[dict[str, Any]] = []
    prev_script = ""
    prev_execution_log = ""
    prev_review = ""
    accumulated_evidence: list[dict[str, Any]] = []
    failure_categories: set[str] = set()
    runtime_payload = _python_runtime_payload()
    log_detail(
        logger,
        "\n%s",
        _format_log_block("[dataset_download_runtime]", runtime_payload),
    )
    for i in range(1, dataset_generation_rounds + 1):
        script_name = f"download{i}.py"
        log_detail(
            logger,
            "\n%s",
            _format_log_block(
                "[dataset_download_round_start]",
                {
                    "dataset_name": candidate.name,
                    "scope_id": candidate.scope_id,
                    "dataset_kind": candidate.dataset_kind,
                    "round": i,
                    "script": script_name,
                    "dataset_dir": dataset_dir,
                },
            ),
        )
        payload = generate_download_code(
            candidate,
            llm,
            execution_log=prev_execution_log,
            previous_review=prev_review,
            previous_script=prev_script,
            previous_evidence=accumulated_evidence,
            failure_categories=sorted(failure_categories),
            timeout_sec=timeout_sec,
        )
        generation_status = str(payload.get("status", "")).strip() or "failed_execution"
        code = str(payload.get("code", "")).strip()
        round_evidence = payload.get("evidence", []) or []
        if isinstance(round_evidence, list):
            accumulated_evidence = _merge_evidence(accumulated_evidence, round_evidence)
        generation_reason = str(payload.get("reason", "")).strip()
        (dataset_dir / script_name).write_text(code, encoding="utf-8")

        # Package installation is outside the Phase 2 contract; reject it before
        # execution and report a repairable failure to the next round.
        sandbox_violations: list[str] = []
        if generation_status == "ready" and code:
            sandbox_violations = Sandbox().scan_phase2_download_violations(code)

        if generation_status == "ready" and code and not sandbox_violations:
            dry = run_download_script(
                dataset_dir,
                script_name=script_name,
                timeout_sec=timeout_sec,
                dry_run=True,
                include_runtime_details=False,
            )
            real = run_download_script(
                dataset_dir,
                script_name=script_name,
                timeout_sec=timeout_sec,
                dry_run=not execute_download,
                include_runtime_details=False,
            )
            real_success = execute_download and real.startswith("returncode=0")
        elif sandbox_violations:
            violation_lines = "\n".join(f"  - {v}" for v in sandbox_violations)
            dry = (
                "skipped: phase-2 sandbox preflight rejected the script "
                "(runtime package management is forbidden in download scripts)"
            )
            real = (
                "skipped: phase-2 sandbox preflight rejected the script.\n"
                "violations:\n"
                f"{violation_lines}"
            )
            real_success = False
            failure_categories.add("runtime_package_install")
            logger.warning(
                "Generated download script for dataset %r (round %d) failed "
                "phase-2 sandbox preflight: %s. The agent will be told via the "
                "`runtime_package_install` failure category and should switch "
                "acquisition strategy.",
                candidate.name,
                i,
                "; ".join(sandbox_violations[:3]),
            )
        else:
            dry = f"generation_status={generation_status}\nreason={generation_reason or '(none)'}"
            real = "skipped: no executable download script generated"
            real_success = False
        data_dir_name = f"data{i}"
        _promote_round_data_dir(dataset_dir, data_dir_name)
        execution_log = "\n".join(["[dry-run]", dry, "", "[real-run]", real])
        # Strip tqdm/gdown progress-bar redraws once at the source so every
        # downstream consumer (failure classifier, reviewer prompt, next-round
        # codegen prompt, on-disk forensic log) sees the same compact log.
        execution_log = _sanitize_execution_log(execution_log)
        failure_categories.update(_classify_download_failures(execution_log))
        missing_modules = _extract_missing_modules(execution_log)
        if missing_modules:
            logger.warning(
                "Generated download script for dataset %r (round %d) imported "
                "package(s) not installed in the fin-pipeline conda environment: "
                "%s. The agent will be told via the `module_not_found` failure "
                "category and should switch acquisition strategy. If you want "
                "these packages available, install them in the active conda env.",
                candidate.name,
                i,
                ", ".join(sorted(missing_modules)),
            )
        evidence_summary = _format_evidence_summary(round_evidence)
        accumulated_evidence_summary = _format_evidence_summary(accumulated_evidence)
        log_detail(
            logger,
            "\n%s",
            _format_log_block(
                "[dataset_download_round_generated]",
                {
                    "dataset_name": candidate.name,
                    "scope_id": candidate.scope_id,
                    "dataset_kind": candidate.dataset_kind,
                    "round": i,
                    "status": generation_status,
                    "round_evidence_count": len(round_evidence),
                    "accumulated_evidence_count": len(accumulated_evidence),
                    "round_evidence": evidence_summary,
                    "accumulated_evidence": accumulated_evidence_summary,
                    "failure_categories": sorted(failure_categories),
                    "reason": generation_reason or "(empty)",
                },
            ),
        )

        review = review_download_script(
            llm=llm,
            dataset_name=candidate.name,
            dataset_description=f"{candidate.name}: {candidate.description}",
            dataset_kind=candidate.dataset_kind,
            script=code,
            execution_log=execution_log,
            evidence=accumulated_evidence,
            generation_status=generation_status,
            generation_reason=generation_reason,
            timeout_sec=timeout_sec,
        )
        rounds.append(
            {
                "round": i,
                "script_name": script_name,
                "code": code,
                "downloaded_dataset_description": str(
                    payload.get("downloaded_dataset_description", "")
                ).strip(),
                "evidence": list(accumulated_evidence),
                "execution_log": execution_log,
                "review": review,
                "real_success": real_success,
                "data_dir_name": data_dir_name,
                "generation_status": generation_status,
                "generation_reason": generation_reason,
                "failure_categories": sorted(failure_categories),
                "round_evidence": list(round_evidence) if isinstance(round_evidence, list) else [],
            }
        )
        log_detail(
            logger,
            "\n%s",
            _format_log_block(
                "[dataset_download_round_review]",
                {
                    "dataset_name": candidate.name,
                    "scope_id": candidate.scope_id,
                    "dataset_kind": candidate.dataset_kind,
                    "round": i,
                    "approved": bool(review.get("approved", False)),
                    "issues": review.get("issues", []) or [],
                    "failure_categories": sorted(failure_categories),
                },
            ),
        )
        approved = bool(review.get("approved", False))
        if approved and real_success:
            log_stage(
                logger,
                "%s   dataset round %d/%d -> approved",
                PHASE2_TAG,
                i,
                dataset_generation_rounds,
            )
        else:
            log_stage(
                logger,
                "%s   dataset round %d/%d -> rejected: %s",
                PHASE2_TAG,
                i,
                dataset_generation_rounds,
                _short_reject_reason_dataset(
                    review,
                    failure_categories=sorted(failure_categories),
                    real_success=real_success,
                    generation_status=generation_status,
                    generation_reason=generation_reason,
                ),
            )
        prev_script = code
        prev_execution_log = execution_log
        prev_review = _feedback_from_review(review)
        if approved and real_success:
            log_detail(
                logger,
                "\n%s",
                _format_log_block(
                    "[dataset_download_round_stop]",
                    {
                        "dataset_name": candidate.name,
                        "scope_id": candidate.scope_id,
                        "dataset_kind": candidate.dataset_kind,
                        "round": i,
                        "reason": "review_approved_and_real_success",
                    },
                ),
            )
            break
    return rounds


def _short_reject_reason_dataset(
    review: dict[str, Any],
    *,
    failure_categories: list[str],
    real_success: bool,
    generation_status: str,
    generation_reason: str,
) -> str:
    if generation_status and generation_status != "ready":
        base = f"generation_status={generation_status}"
        if generation_reason:
            return truncate_oneline(f"{base} ({generation_reason})")
        return truncate_oneline(base)
    if not real_success and failure_categories:
        return truncate_oneline("download failed: " + ",".join(failure_categories))
    issues = review.get("issues") or []
    if issues:
        return truncate_oneline(str(issues[0]))
    if not real_success:
        return "download did not return code 0"
    return "review not approved"


def _rewrite_reward_rounds(
    *,
    reward_dir: Path,
    reward: RewardCandidate,
    llm: Any | None,
    rewards_root: Path | None = None,
    reward_generation_rounds: int,
) -> list[dict[str, Any]]:
    if reward_generation_rounds < 1:
        raise ValueError("reward_generation_rounds must be >= 1")
    rounds: list[dict[str, Any]] = []
    prev_script = ""
    prev_execution_log = ""
    prev_review = ""
    for i in range(1, reward_generation_rounds + 1):
        script_name = f"reward_fn{i}.py"
        payload = generate_reward_code(
            reward,
            llm,
            execution_log=prev_execution_log,
            previous_review=prev_review,
            previous_script=prev_script,
        )
        # ``generate_reward_code`` returns a dict but some older tests
        # stub it with a plain string. Normalise both shapes.
        if isinstance(payload, str):
            payload = {"code": payload}
        code = str(payload.get("code", ""))
        (reward_dir / script_name).write_text(code, encoding="utf-8")

        # AST pre-flight: check the code conforms to the reward_fn contract
        # (RULES R1-R6 from prompts.contract_rules) BEFORE burning a pytest
        # run. On violation, build an execution_log that surfaces the
        # exact rule failures so the reviewer can echo them as round-N+1
        # feedback.
        violations = validate_reward_contract(code)
        if violations:
            execution_log = (
                "Contract violations (RULES R1-R6) — must be fixed in next round:\n"
                + "\n".join(f"- {v}" for v in violations)
                + "\n\n[pytest skipped: contract rejected]"
            )
        else:
            execution_log = run_reward_script(
                reward_dir, script_name=script_name, rewards_root=rewards_root
            )
        review = review_reward_code(
            llm,
            code,
            reward_name=reward.name,
            reward_description=reward.description,
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
            }
        )
        approved = bool(review.get("approved", False))
        if approved:
            log_stage(
                logger,
                "%s   reward round %d/%d -> approved",
                PHASE3_TAG,
                i,
                reward_generation_rounds,
            )
        else:
            log_stage(
                logger,
                "%s   reward round %d/%d -> rejected: %s",
                PHASE3_TAG,
                i,
                reward_generation_rounds,
                _short_reject_reason_reward(review, violations),
            )
        prev_script = code
        prev_execution_log = execution_log
        prev_review = _feedback_from_review(review)
        if approved:
            break
    return rounds


def _short_reject_reason_reward(
    review: dict[str, Any],
    violations: list[str] | None,
) -> str:
    issues = review.get("issues") or []
    if issues:
        return truncate_oneline(str(issues[0]))
    if violations:
        return truncate_oneline(violations[0])
    analysis = str(review.get("analysis", "")).strip()
    if analysis:
        return truncate_oneline(analysis)
    return "no specific issue surfaced"


def _write_download_round_logs(dataset_dir: Path, rounds: list[dict[str, Any]]) -> None:
    for r in rounds:
        round_idx = int(r.get("round", 0))
        if round_idx <= 0:
            continue
        path = dataset_dir / f"download{round_idx}.log"
        lines = [
            f"[round {round_idx}]",
            f"script={r.get('script_name', '')}",
            f"generation_status={r.get('generation_status', '')}",
            f"generation_reason={str(r.get('generation_reason', '')).strip() or '(empty)'}",
            f"real_success={bool(r.get('real_success', False))}",
            f"failure_categories={','.join(r.get('failure_categories', [])) or '(none)'}",
            "",
            "[round_evidence]",
            _format_evidence_summary(r.get("round_evidence", []) or []),
            "",
            "[accumulated_evidence]",
            _format_evidence_summary(r.get("evidence", []) or []),
            "",
            "[execution_log]",
            str(r.get("execution_log", "")).strip(),
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        json_path = dataset_dir / f"download{round_idx}.json"
        json_path.write_text(
            json.dumps(
                {
                    "round": round_idx,
                    "script_name": str(r.get("script_name", "")),
                    "generation_status": str(r.get("generation_status", "")),
                    "generation_reason": str(r.get("generation_reason", "")),
                    "downloaded_dataset_description": str(
                        r.get("downloaded_dataset_description", "")
                    ),
                    "real_success": bool(r.get("real_success", False)),
                    "failure_categories": list(r.get("failure_categories", [])),
                    "round_evidence": list(r.get("round_evidence", [])),
                    "accumulated_evidence": list(r.get("evidence", [])),
                    "review": r.get("review", {}),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def _feedback_from_review(review: dict[str, Any]) -> str:
    analysis = str(review.get("analysis", "")).strip()
    issues = [str(x) for x in review.get("issues", []) if str(x).strip()]
    suggestions = [str(x) for x in review.get("suggestions", []) if str(x).strip()]
    lines = []
    if analysis:
        lines.append("Reviewer analysis:")
        lines.append(analysis)
    if issues:
        lines.append("Issues:")
        lines.extend(f"- {x}" for x in issues)
    if suggestions:
        lines.append("Suggestions:")
        lines.extend(f"- {x}" for x in suggestions)
    return "\n".join(lines).strip()


def _promote_round_data_dir(dataset_dir: Path, target_name: str) -> None:
    src = dataset_dir / "data"
    if not src.exists():
        return
    dst = dataset_dir / target_name
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)

    # On Windows the download subprocess may not have fully released file
    # handles yet, so retry the rename a few times before falling back to
    # a copy-then-delete approach.
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            src.rename(dst)
            return
        except PermissionError as exc:
            last_err = exc
            logger.debug(
                "rename %s -> %s failed (attempt %d): %s",
                src,
                dst,
                attempt + 1,
                exc,
            )
            time.sleep(1)

    # Fallback: copy tree then remove source.
    logger.warning(
        "rename failed after retries, falling back to copy: %s -> %s (%s)",
        src,
        dst,
        last_err,
    )
    shutil.copytree(src, dst)
    shutil.rmtree(src, ignore_errors=True)


def _merge_evidence(
    existing: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(existing)
    seen = {
        (
            str(item.get("source_type", "")).strip(),
            str(item.get("title", "")).strip(),
            str(item.get("url", "")).strip(),
            str(item.get("note", "")).strip(),
        )
        for item in existing
        if isinstance(item, dict)
    }
    for item in new_items:
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("source_type", "")).strip(),
            str(item.get("title", "")).strip(),
            str(item.get("url", "")).strip(),
            str(item.get("note", "")).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _format_evidence_summary(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "(none)"
    lines: list[str] = []
    for idx, item in enumerate(evidence, start=1):
        if not isinstance(item, dict):
            lines.append(f"{idx}. (invalid evidence item)")
            continue
        source_type = str(item.get("source_type", "")).strip() or "(unknown_source)"
        title = str(item.get("title", "")).strip() or "(untitled)"
        url = str(item.get("url", "")).strip() or "(no_url)"
        note = str(item.get("note", "")).strip() or "(no_note)"
        lines.append(f"{idx}. [{source_type}] {title}")
        lines.append(f"   url: {url}")
        lines.append(f"   note: {note}")
    return "\n".join(lines)


# tqdm/gdown progress-bar lines: `<spaces><digits>%|<bar>|<quantity>
# [<time>, <rate>]`. Universal-newline mode turns each `\r` into a fresh
# line, so a single gdown download can land hundreds in stderr and drown
# the real signal in the reviewer's input. Bar contents never contain
# `|`, so `[^|]*` is safe.
_PROGRESS_BAR_RE = re.compile(r"^\s*\d+%\|[^|]*\|.*\]\s*$")


def _sanitize_execution_log(log: str) -> str:
    """Strip tqdm/gdown-style progress-bar lines from an execution log.

    Consecutive stripped lines are collapsed into a single
    ``[N progress-bar lines stripped]`` marker so the LLM still sees that
    download activity occurred, without paying tokens for hundreds of bar
    redraws. All other lines (URLs, ``Downloading...``/``To:`` headers,
    error tracebacks, exit codes) are preserved verbatim — the failure
    classifier and reviewer rely on those.
    """
    out: list[str] = []
    stripped_run = 0
    for line in log.splitlines():
        if _PROGRESS_BAR_RE.match(line):
            stripped_run += 1
            continue
        if stripped_run:
            out.append(f"[{stripped_run} progress-bar line(s) stripped]")
            stripped_run = 0
        out.append(line)
    if stripped_run:
        out.append(f"[{stripped_run} progress-bar line(s) stripped]")
    return "\n".join(out)


_MODULE_NOT_FOUND_RE = re.compile(
    r"no module named ['\"]([^'\"]+)['\"]", re.IGNORECASE
)

# Generated scripts phrase credential failures inconsistently. Match
# credential-like environment names near common missing-value terms.
_ENV_VAR_NAME_PATTERN = (
    # Prefix is at least 1 char (so HF_TOKEN, AWS_KEY etc. match) plus the
    # credential suffix. _USER/_PWD/_API kept short on purpose because they
    # are common envvar tails.
    r"[A-Z][A-Z0-9_]*(?:_KEY|_TOKEN|_SECRET|_API|_API_KEY|_PASSWORD|_USERNAME|_USER|_PWD)"
)
_MISSING_LANG = (
    r"missing|not set|unset|please set|please provide|required|set the environment "
    r"variable|set a valid"
)
_MISSING_ENV_VAR_RE = re.compile(
    rf"(?:{_MISSING_LANG})[^\n]{{0,80}}?{_ENV_VAR_NAME_PATTERN}"
    rf"|{_ENV_VAR_NAME_PATTERN}[^\n]{{0,80}}?(?:{_MISSING_LANG})",
    re.IGNORECASE,
)


def _classify_download_failures(execution_log: str) -> set[str]:
    lowered = execution_log.lower()
    labels: set[str] = set()
    if "http 404" in lowered or "http_error:404" in lowered or "status_code=404" in lowered:
        labels.add("404")
    if "http 403" in lowered or "http_error:403" in lowered or "status_code=403" in lowered:
        labels.add("403")
    # subprocess.TimeoutExpired prints "timed out after N seconds"; the
    # executor's own marker prepends "timeout:". Match both shapes so the
    # classifier is robust regardless of which path triggered the kill.
    if "timeout" in lowered or "timed out" in lowered:
        labels.add("timeout")
    if (
        "missing required env vars" in lowered
        or "missing credential" in lowered
        or _MISSING_ENV_VAR_RE.search(execution_log) is not None
    ):
        labels.add("missing_env_var")
    if "downloaded file is empty" in lowered or "data/ is empty" in lowered:
        labels.add("empty_payload")
    if "<html" in lowered or "text/html" in lowered:
        labels.add("html_not_file")
    if "zipfile.badzipfile" in lowered or "badzipfile" in lowered:
        labels.add("bad_archive")
    if "modulenotfounderror" in lowered or "no module named" in lowered:
        labels.add("module_not_found")
    return labels


def _extract_missing_modules(execution_log: str) -> set[str]:
    """Pull module names out of `ModuleNotFoundError: No module named 'X'` lines.

    Returns the top-level package name (e.g. ``foo`` from ``foo.bar.baz``) so
    the warning matches what a user would `pip install`.
    """
    return {match.split(".", 1)[0] for match in _MODULE_NOT_FOUND_RE.findall(execution_log)}
