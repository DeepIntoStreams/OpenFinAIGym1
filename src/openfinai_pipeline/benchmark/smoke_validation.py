"""Pre- and post-install smoke gates for assembled task evaluators.

Both gates score zero predictions against held-out or reference data.
The dry-fire gate validates staged assets; the installed gate also checks
bundle layout and can emit a failure sidecar.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import logging
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from openfinai_pipeline.utils.logging import log_detail, truncate_oneline

logger = logging.getLogger(__name__)


def short_smoke_reason(validation: dict) -> str:
    """Return the first validation error as a compact stage marker."""
    errors = list(validation.get("errors") or [])
    if not errors:
        return "unknown"
    first = truncate_oneline(str(errors[0]))
    if len(errors) > 1:
        return f"{first} (+{len(errors) - 1} more)"
    return first

_DRY_FIRE_SUBDIR = ".dry_fire"
_DRY_FIRE_REWARD_FILE = "_reward.json"
_REQUIRED_RELATIVE_PATHS: tuple[str, ...] = (
    "instruction.md",
    "task.toml",
    "tests/test.sh",
    "environment/Dockerfile",
    "environment/data/evaluator.py",
    "environment/data/task.py",
    "environment/data/manifest.json",
)


def validate_installed_task_bundle(
    task_dir: Path,
    *,
    scope_id: str,
    task_id: str,
    staging_dir: Path | None = None,
) -> dict:
    """Check an installed bundle's layout and zero-prediction score.

    ``staging_dir`` redirects the reward sidecar outside the installed
    bundle. The result includes identifiers, errors, checked paths, and
    the smoke result.
    """
    errors: list[str] = []
    layout_checked: list[str] = []
    for rel in _REQUIRED_RELATIVE_PATHS:
        layout_checked.append(rel)
        if not (task_dir / rel).exists():
            errors.append(f"missing:{rel}")

    # Static datasets require loader and reward assets; trading does not.
    manifest_path = task_dir / "environment" / "data" / "manifest.json"
    manifest_payload: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest_payload = {}
    interaction_model = str(manifest_payload.get("interaction_model", "")).strip()
    if interaction_model in ("forecasting", "generative"):
        for rel in (
            "environment/data/dataset.h5",
            "environment/data/task_rewards.py",
            "environment/data/load.py",
        ):
            layout_checked.append(rel)
            if not (task_dir / rel).exists():
                errors.append(f"missing:{rel}")
        if manifest_payload.get("has_held_out_test_gt", True):
            rel = "environment/eval-data/test_ground_truth.h5"
            layout_checked.append(rel)
            if not (task_dir / rel).exists():
                errors.append(f"missing:{rel}")

    smoke: dict = {"ran": False, "errors": []}
    if not errors:
        smoke = _smoke_validate_installed_bundle(task_dir, staging_dir=staging_dir)
        errors.extend(smoke.get("errors", []))

    return {
        "task": f"{scope_id}__{task_id}",
        "scope_id": scope_id,
        "task_id": task_id,
        "valid": not errors,
        "errors": errors,
        "layout_checked": layout_checked,
        "smoke": smoke,
    }


def write_validation_failure_sidecar(
    task_dir: Path,
    *,
    tasks_root: Path,
    scope_id: str,
    task_id: str,
    result: dict,
) -> Path:
    """Write the latest failure sidecar using portable path strings."""
    try:
        rel_task_dir = (
            task_dir.resolve().relative_to(tasks_root.resolve()).as_posix()
        )
    except ValueError:
        rel_task_dir = task_dir.resolve().as_posix()
    payload = {
        "schema_version": 1,
        "phase_tag": "phase4:benchmark",
        "scope_id": scope_id,
        "task_id": task_id,
        "task_dir": rel_task_dir,
        "validated_at_utc": datetime.now(tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "errors": list(result.get("errors", [])),
        "smoke": result.get("smoke", {}),
    }
    sidecar = task_dir / "validation_failed.json"
    sidecar.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return sidecar


def _load_module_from_path(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zeros_like_gt(gt: Any):
    """Build zeros matching an array or concatenated symbol panel."""
    import numpy as np

    if isinstance(gt, dict):
        gt_concat = np.concatenate(
            [np.asarray(v, dtype=np.float64) for v in gt.values()],
            axis=0,
        )
        return np.zeros_like(gt_concat)
    return np.zeros_like(np.asarray(gt, dtype=np.float64))


def _smoke_validate_installed_bundle(
    task_dir: Path,
    *,
    staging_dir: Path | None = None,
) -> dict:
    """Run the installed evaluator with zero predictions.

    Held-out tasks read ``eval-data/test_ground_truth.h5``; unconditional
    tasks use the loader's ``reference`` value. ``staging_dir`` redirects
    ``validation/reward.json`` outside the installed bundle.
    """
    errors: list[str] = []
    data_dir = task_dir / "environment" / "data"
    eval_data_dir = task_dir / "environment" / "eval-data"
    evaluator_path = data_dir / "evaluator.py"
    load_path = data_dir / "load.py"

    base_result: dict[str, Any] = {
        "ran": False,
        "score_succeeded": False,
        "reward_path": "",
        "scores": {},
        "evaluator_path": evaluator_path.as_posix(),
        "data_dir": data_dir.as_posix(),
        "eval_data_dir": eval_data_dir.as_posix(),
    }

    if not evaluator_path.exists() or not load_path.exists():
        return {**base_result, "errors": ["missing_verifier_assets"]}

    # Parse failures use the held-out default.
    has_held_out = True
    manifest_path = data_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        has_held_out = bool(manifest.get("has_held_out_test_gt", True))
    except (OSError, json.JSONDecodeError):
        pass

    ground_truth: Any
    if has_held_out:
        test_gt_path = eval_data_dir / "test_ground_truth.h5"
        if not test_gt_path.exists():
            return {
                **base_result,
                "errors": [f"missing:{test_gt_path.as_posix()}"],
            }
        try:
            from openfinai_pipeline.benchmark.install.runtime_loader import (
                read_test_gt_h5,
            )

            ground_truth = read_test_gt_h5(test_gt_path)
        except Exception as exc:  # noqa: BLE001 — surface as smoke error
            return {
                **base_result,
                "errors": [
                    f"read_test_ground_truth:{type(exc).__name__}:{exc}"
                ],
            }
    else:
        load_unique = f"_smoke_loader_{abs(hash(str(load_path)))}"
        try:
            loader_module = _load_module_from_path(load_unique, load_path)
            loader_out = loader_module.load(data_dir)
        except Exception as exc:  # noqa: BLE001
            return {
                **base_result,
                "errors": [f"load_call:{type(exc).__name__}:{exc}"],
            }
        finally:
            sys.modules.pop(load_unique, None)
        if not isinstance(loader_out, dict) or "reference" not in loader_out:
            keys = (
                sorted(loader_out.keys())
                if isinstance(loader_out, dict)
                else type(loader_out).__name__
            )
            return {
                **base_result,
                "errors": [
                    "load_no_reference: unconditional generative loader "
                    f"must return a 'reference' key; got keys={keys}"
                ],
            }
        ground_truth = loader_out["reference"]
        if ground_truth is None:
            return {**base_result, "errors": ["load_reference_none"]}

    try:
        predictions = _zeros_like_gt(ground_truth)
    except Exception as exc:  # noqa: BLE001
        return {
            **base_result,
            "errors": [f"zeros_alloc:{type(exc).__name__}:{exc}"],
        }

    eval_unique = f"_smoke_evaluator_{abs(hash(str(evaluator_path)))}"
    # Let the evaluator create the reward directory; remove it if left empty.
    reward_root = staging_dir if staging_dir is not None else task_dir
    reward_output = reward_root / "validation" / "reward.json"
    scores: Any = {}
    score_succeeded = False
    captured_stdout = io.StringIO()
    try:
        evaluator_module = _load_module_from_path(eval_unique, evaluator_path)
        evaluator_cls = _find_evaluator_class(evaluator_module)
        if evaluator_cls is None:
            errors.append(
                "evaluator_class_not_found: no BaseEvaluator subclass exposed by "
                f"{evaluator_path.as_posix()}"
            )
        else:
            evaluator = evaluator_cls()
            with contextlib.redirect_stdout(captured_stdout):
                scores = evaluator.score(
                    predictions=predictions,
                    ground_truth=ground_truth,
                    reward_output=reward_output,
                    data_dir=data_dir,
                    eval_data_dir=eval_data_dir,
                )
            score_succeeded = True
    except Exception as exc:  # noqa: BLE001 — surface as smoke error
        errors.append(f"smoke_validation:{type(exc).__name__}:{exc}")
    finally:
        sys.modules.pop(eval_unique, None)
        captured_text = captured_stdout.getvalue().strip()
        if captured_text:
            log_detail(logger, "smoke score stdout:\n%s", captured_text)
        validation_dir = reward_output.parent
        if validation_dir.exists() and not any(validation_dir.iterdir()):
            try:
                validation_dir.rmdir()
            except OSError:
                pass

    if not isinstance(scores, dict):
        errors.append(
            f"score_return_type: expected dict, got {type(scores).__name__}"
        )
        scores = {}
    if score_succeeded:
        if not reward_output.exists():
            errors.append("reward_json_not_written")
        if "reward" not in scores:
            errors.append("reward_key_missing")

    return {
        "ran": True,
        "score_succeeded": score_succeeded,
        "reward_path": reward_output.as_posix() if reward_output.exists() else "",
        "scores": scores if isinstance(scores, dict) else {},
        "evaluator_path": evaluator_path.as_posix(),
        "data_dir": data_dir.as_posix(),
        "eval_data_dir": eval_data_dir.as_posix(),
        "errors": errors,
    }

def _materialize_dry_fire_layout(
    *,
    staging_dir: Path,
    data_dir: Path,
    dry_fire_dir: Path,
) -> list[str]:
    """Mirror the staged loader and dataset into an install-like directory."""
    errors: list[str] = []
    staged_load = staging_dir / "load.py"
    if not staged_load.exists():
        errors.append(f"missing:staging/load.py at {staged_load.as_posix()}")
        return errors
    if not data_dir.exists() or not data_dir.is_dir():
        errors.append(f"missing:data_dir at {data_dir.as_posix()}")
        return errors

    if dry_fire_dir.exists():
        shutil.rmtree(str(dry_fire_dir), ignore_errors=True)
    dry_fire_dir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(str(staged_load), str(dry_fire_dir / "load.py"))
    except OSError as exc:
        errors.append(f"copy_load:{exc}")
        return errors

    try:
        for src in data_dir.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(data_dir)
            dst = dry_fire_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
    except OSError as exc:
        errors.append(f"copy_payload:{exc}")
        return errors

    return errors


def _find_evaluator_class(module: ModuleType) -> type | None:
    """Find the assembled ``BaseEvaluator`` subclass, with a name fallback."""
    try:
        from openfinai_pipeline.benchmark.contracts import BaseEvaluator
    except Exception:  # noqa: BLE001 — best-effort
        BaseEvaluator = None  # type: ignore[assignment]

    candidates: list[type] = []
    for attr in dir(module):
        if attr.startswith("_"):
            continue
        obj = getattr(module, attr, None)
        if not isinstance(obj, type):
            continue
        if BaseEvaluator is not None and obj is BaseEvaluator:
            continue
        if obj.__module__ != module.__name__:
            continue  # re-exported symbol
        if BaseEvaluator is not None and issubclass(obj, BaseEvaluator):
            candidates.append(obj)
            continue
        if attr.endswith("Evaluator") and hasattr(obj, "score"):
            candidates.append(obj)

    if not candidates:
        return None
    for cls in candidates:
        if hasattr(cls, "score") and getattr(cls, "REWARD_NAMES", None):
            return cls
    return candidates[0]


def _empty_dry_fire_result(
    *,
    staging_dir: Path,
    data_dir: Path,
    dry_fire_dir: Path,
    errors: list[str],
) -> dict:
    return {
        "ran": False,
        "ok": False,
        "errors": list(errors),
        "evaluator_path": (staging_dir / "evaluator.py").as_posix(),
        "load_path": (staging_dir / "load.py").as_posix(),
        "data_dir": data_dir.as_posix() if data_dir is not None else "",
        "dry_fire_dir": dry_fire_dir.as_posix(),
        "reward_path": "",
        "scores": {},
        "score_keys": [],
        "prediction_blind": [],
        "perfect_scores": {},
    }


def dry_fire_evaluator(
    *,
    staging_dir: Path,
    data_dir: Path,
    resolved_reward_names: list[str],
) -> dict:
    """Run the staged evaluator with zero and perfect predictions.

    ``staging_dir`` must contain ``evaluator.py`` and ``load.py``;
    ``data_dir`` supplies the dataset payload. ``resolved_reward_names``
    must contain slugified score-dict keys, each of which must resolve to a
    finite score. Exceptions are returned in ``errors`` rather than raised.
    """
    dry_fire_dir = staging_dir / _DRY_FIRE_SUBDIR
    evaluator_path = staging_dir / "evaluator.py"

    if not evaluator_path.exists():
        return _empty_dry_fire_result(
            staging_dir=staging_dir,
            data_dir=data_dir,
            dry_fire_dir=dry_fire_dir,
            errors=[f"missing:staging/evaluator.py at {evaluator_path.as_posix()}"],
        )

    layout_errors = _materialize_dry_fire_layout(
        staging_dir=staging_dir,
        data_dir=data_dir,
        dry_fire_dir=dry_fire_dir,
    )
    if layout_errors:
        return _empty_dry_fire_result(
            staging_dir=staging_dir,
            data_dir=data_dir,
            dry_fire_dir=dry_fire_dir,
            errors=layout_errors,
        )

    # Keep the module importable when NumPy is absent.
    try:
        import numpy as np
    except ImportError as exc:
        return _empty_dry_fire_result(
            staging_dir=staging_dir,
            data_dir=data_dir,
            dry_fire_dir=dry_fire_dir,
            errors=[f"numpy_import:{exc}"],
        )

    load_unique_name = (
        f"_dry_fire_loader_{abs(hash(str(dry_fire_dir / 'load.py')))}"
    )
    try:
        loader_module = _load_module_from_path(
            load_unique_name, dry_fire_dir / "load.py"
        )
    except Exception as exc:  # noqa: BLE001 — surface as gate error
        return _empty_dry_fire_result(
            staging_dir=staging_dir,
            data_dir=data_dir,
            dry_fire_dir=dry_fire_dir,
            errors=[f"load_module_import:{type(exc).__name__}:{exc}"],
        )

    try:
        loader_out = loader_module.load(dry_fire_dir)
    except Exception as exc:  # noqa: BLE001
        return _empty_dry_fire_result(
            staging_dir=staging_dir,
            data_dir=data_dir,
            dry_fire_dir=dry_fire_dir,
            errors=[f"load_call:{type(exc).__name__}:{exc}"],
        )
    finally:
        sys.modules.pop(load_unique_name, None)

    if not isinstance(loader_out, dict):
        return _empty_dry_fire_result(
            staging_dir=staging_dir,
            data_dir=data_dir,
            dry_fire_dir=dry_fire_dir,
            errors=[
                "load_return_shape: load() must return a dict; "
                f"got {type(loader_out).__name__}"
            ],
        )

    if "reference" in loader_out:
        ground_truth = loader_out["reference"]
        if ground_truth is None:
            return _empty_dry_fire_result(
                staging_dir=staging_dir,
                data_dir=data_dir,
                dry_fire_dir=dry_fire_dir,
                errors=["load_reference_none: 'reference' key is None"],
            )
    elif "test" in loader_out and isinstance(loader_out["test"], dict):
        ground_truth = loader_out["test"].get("ground_truth")
        if ground_truth is None:
            return _empty_dry_fire_result(
                staging_dir=staging_dir,
                data_dir=data_dir,
                dry_fire_dir=dry_fire_dir,
                errors=[
                    "load_test_ground_truth_none: forecasting/conditional "
                    "generative loader returned test.ground_truth=None; "
                    "dry-fire requires a non-None target array. The "
                    "install slicer omits this from the agent dataset.h5 "
                    "but the LLM loader's pre-install output must contain "
                    "it for the slicer to demux."
                ],
            )
    else:
        return _empty_dry_fire_result(
            staging_dir=staging_dir,
            data_dir=data_dir,
            dry_fire_dir=dry_fire_dir,
            errors=[
                "load_return_shape: load() output must contain either a "
                "'reference' key (unconditional generative) or a 'test' "
                f"bundle (B-shape); got keys={sorted(loader_out.keys())}"
            ],
        )

    try:
        if isinstance(ground_truth, dict):
            concatenated = np.concatenate(
                [np.asarray(v, dtype=np.float64) for v in ground_truth.values()],
                axis=0,
            )
            gt_array = concatenated
        else:
            gt_array = np.asarray(ground_truth, dtype=np.float64)
        predictions = np.zeros_like(gt_array)
    except Exception as exc:  # noqa: BLE001
        return _empty_dry_fire_result(
            staging_dir=staging_dir,
            data_dir=data_dir,
            dry_fire_dir=dry_fire_dir,
            errors=[f"zeros_alloc:{type(exc).__name__}:{exc}"],
        )

    eval_unique_name = f"_dry_fire_evaluator_{abs(hash(str(evaluator_path)))}"
    try:
        evaluator_module = _load_module_from_path(eval_unique_name, evaluator_path)
    except Exception as exc:  # noqa: BLE001
        return _empty_dry_fire_result(
            staging_dir=staging_dir,
            data_dir=data_dir,
            dry_fire_dir=dry_fire_dir,
            errors=[f"evaluator_module_import:{type(exc).__name__}:{exc}"],
        )

    evaluator_cls = _find_evaluator_class(evaluator_module)
    if evaluator_cls is None:
        sys.modules.pop(eval_unique_name, None)
        return _empty_dry_fire_result(
            staging_dir=staging_dir,
            data_dir=data_dir,
            dry_fire_dir=dry_fire_dir,
            errors=[
                "evaluator_class_not_found: no BaseEvaluator subclass exposed by "
                f"{evaluator_path.as_posix()}"
            ],
        )

    reward_output = dry_fire_dir / _DRY_FIRE_REWARD_FILE
    agent_output_dir = dry_fire_dir / "_agent_out"
    agent_output_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    scores: Any = {}
    perfect_scores: Any = {}
    ran = False
    perfect_ran = False
    # Capture evaluator tables so they respect the logging policy.
    captured_stdout = io.StringIO()
    perfect_captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_stdout):
            evaluator = evaluator_cls()
            scores = evaluator.score(
                predictions=predictions,
                ground_truth=gt_array,
                reward_output=reward_output,
                agent_output_dir=agent_output_dir,
                data_dir=dry_fire_dir,
            )
        ran = True
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=5)
        errors.append(
            f"score_raised:{type(exc).__name__}:{exc} | tb:{tb.splitlines()[-2] if tb else ''}"
        )
    finally:
        captured_text = captured_stdout.getvalue().strip()
        if captured_text:
            log_detail(logger, "dry_fire stdout:\n%s", captured_text)

    # Equal finite scores for zero and perfect predictions indicate a blind metric.
    if ran and isinstance(scores, dict):
        try:
            with contextlib.redirect_stdout(perfect_captured):
                evaluator_perfect = evaluator_cls()
                perfect_scores = evaluator_perfect.score(
                    predictions=gt_array,
                    ground_truth=gt_array,
                    reward_output=None,
                    agent_output_dir=agent_output_dir,
                    data_dir=dry_fire_dir,
                )
            perfect_ran = True
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc(limit=5)
            errors.append(
                "score_perfect_raised:"
                f"{type(exc).__name__}:{exc} | "
                f"tb:{tb.splitlines()[-2] if tb else ''}"
            )
        finally:
            perfect_text = perfect_captured.getvalue().strip()
            if perfect_text:
                log_detail(
                    logger,
                    "dry_fire perfect_pred stdout:\n%s",
                    perfect_text,
                )

    sys.modules.pop(eval_unique_name, None)

    if not isinstance(scores, dict):
        errors.append(
            f"score_return_type: expected dict, got {type(scores).__name__}"
        )
        scores = {}
    if not isinstance(perfect_scores, dict):
        perfect_scores = {}

    # Non-finite metrics are classified below, not as prediction-blind.
    prediction_blind: list[str] = []
    if perfect_ran:
        zeros_per_reward_errors: dict[str, Any] = (
            (scores.get("_errors") or {}) if isinstance(scores, dict) else {}
        )
        perfect_per_reward_errors: dict[str, Any] = perfect_scores.get(
            "_errors"
        ) or {}
        if isinstance(perfect_per_reward_errors, dict):
            for rk, msg in perfect_per_reward_errors.items():
                if rk in zeros_per_reward_errors:
                    continue  # already surfaced by zeros-side reward_error
                errors.append(f"reward_perfect_error:{rk}:{msg}")
        for name in resolved_reward_names or []:
            if name not in scores or name not in perfect_scores:
                continue
            if name in zeros_per_reward_errors or (
                isinstance(perfect_per_reward_errors, dict)
                and name in perfect_per_reward_errors
            ):
                continue
            try:
                z_val = float(scores[name])
                p_val = float(perfect_scores[name])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(z_val) and np.isfinite(p_val)):
                continue
            if z_val == p_val:
                prediction_blind.append(name)

    if prediction_blind:
        errors.append(
            f"prediction_blind_metrics:{sorted(prediction_blind)}"
        )

    if ran and isinstance(scores, dict):
        global_err = scores.get("_error")
        if global_err:
            errors.append(f"score_error:{global_err}")
        per_reward_errors = scores.get("_errors") or {}
        if isinstance(per_reward_errors, dict) and per_reward_errors:
            for rk, msg in per_reward_errors.items():
                errors.append(f"reward_error:{rk}:{msg}")
        reward_total = scores.get("reward")
        if reward_total is None:
            errors.append("score_missing_reward_key: aggregator 'reward' absent")
        else:
            try:
                if not np.isfinite(float(reward_total)):
                    errors.append(
                        f"score_reward_non_finite: reward={reward_total!r}"
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"score_reward_non_numeric: reward={reward_total!r}"
                )
        # Every requested score key must be present and finite.
        for name in resolved_reward_names or []:
            if name not in scores:
                errors.append(f"reward_dispatch_missing:{name}")
                continue
            val = scores.get(name)
            try:
                fval = float(val)
            except (TypeError, ValueError):
                errors.append(f"reward_value_non_numeric:{name}:{val!r}")
                continue
            if not np.isfinite(fval):
                errors.append(f"reward_value_non_finite:{name}:{val!r}")
        if not reward_output.exists():
            errors.append("reward_json_not_written")
        else:
            try:
                json.loads(reward_output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"reward_json_invalid:{exc}")

    return {
        "ran": ran,
        "ok": ran and not errors,
        "errors": errors,
        "evaluator_path": evaluator_path.as_posix(),
        "load_path": (staging_dir / "load.py").as_posix(),
        "data_dir": data_dir.as_posix(),
        "dry_fire_dir": dry_fire_dir.as_posix(),
        "reward_path": (
            reward_output.as_posix() if reward_output.exists() else ""
        ),
        "scores": scores if isinstance(scores, dict) else {},
        "score_keys": (
            sorted(scores.keys()) if isinstance(scores, dict) else []
        ),
        "prediction_blind": prediction_blind,
        "perfect_scores": perfect_scores if isinstance(perfect_scores, dict) else {},
    }


def write_dry_fire_failure_sidecar(
    staging_dir: Path,
    *,
    scope_id: str,
    task_id: str,
    result: dict,
) -> Path:
    """Write the latest dry-fire failure sidecar to ``staging_dir``."""
    payload = {
        "schema_version": 1,
        "phase_tag": "phase4:benchmark",
        "gate": "dry_fire",
        "scope_id": scope_id,
        "task_id": task_id,
        "staging_dir": staging_dir.resolve().as_posix(),
        "validated_at_utc": datetime.now(tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "errors": list(result.get("errors", [])),
        "ran": bool(result.get("ran", False)),
        "evaluator_path": str(result.get("evaluator_path", "")),
        "load_path": str(result.get("load_path", "")),
        "data_dir": str(result.get("data_dir", "")),
        "dry_fire_dir": str(result.get("dry_fire_dir", "")),
        "reward_path": str(result.get("reward_path", "")),
        "score_keys": list(result.get("score_keys", [])),
    }
    staging_dir.mkdir(parents=True, exist_ok=True)
    sidecar = staging_dir / "dry_fire_failed.json"
    sidecar.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return sidecar
