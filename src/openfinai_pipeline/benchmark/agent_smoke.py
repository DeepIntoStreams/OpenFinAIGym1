"""Smoke-test generated tasks through their public and direct evaluator paths."""

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
from typing import Any, Dict

from openfinai_pipeline.benchmark.smoke_validation import (
    _find_evaluator_class,
)
from openfinai_pipeline.utils.logging import log_detail

logger = logging.getLogger(__name__)

_SMOKE_SUBDIR = ".agent_smoke"
_SMOKE_FAILURE_SIDECAR = "agent_smoke_failed.json"

# Purge staged modules around each run to prevent cross-round reuse.
_BARE_NAME_MODULES_TO_CLEAN = ("evaluator", "load", "task", "task_rewards")
_HASHED_PREFIXES_TO_CLEAN = ("_smoke_task_", "_smoke_evaluator_", "_smoke_loader_")

# Maximum failure context returned to the generator.
_LLM_FEEDBACK_BYTE_BUDGET = 8000


def agent_smoke_validate(
    *,
    staging_dir: Path,
    data_dir: Path,
    interaction_model: str,
) -> dict:
    """Run both smoke modes and return a non-raising structured result."""
    smoke_dir = staging_dir / _SMOKE_SUBDIR

    base_result: Dict[str, Any] = {
        "ran": False,
        "ok": False,
        "errors": [],
        "modes": {},
        "smoke_dir": smoke_dir.as_posix(),
        "interaction_model": interaction_model,
    }

    # Unsupported task families are outside this gate.
    if interaction_model not in ("forecasting", "generative"):
        return {
            **base_result,
            "ran": False,
            "ok": True,
            "errors": [],
        }

    required = (
        staging_dir / "task.py",
        staging_dir / "evaluator.py",
        staging_dir / "load.py",
        staging_dir / "task_rewards.py",
    )
    missing = [p for p in required if not p.exists()]
    if missing:
        rels = [p.relative_to(staging_dir).as_posix() for p in missing]
        return {
            **base_result,
            "ran": False,
            "ok": False,
            "errors": [f"missing_staged_artefact:{r}" for r in rels],
        }

    if not data_dir.exists() or not data_dir.is_dir():
        return {
            **base_result,
            "ran": False,
            "ok": False,
            "errors": [f"missing:data_dir at {data_dir.as_posix()}"],
        }

    layout_errors = _materialize_smoke_layout(
        staging_dir=staging_dir,
        data_dir=data_dir,
        smoke_dir=smoke_dir,
    )
    if layout_errors:
        return {
            **base_result,
            "ran": False,
            "ok": False,
            "errors": layout_errors,
        }

    try:
        import numpy as np
    except ImportError as exc:
        return {
            **base_result,
            "ran": False,
            "ok": False,
            "errors": [f"numpy_import:{exc}"],
        }

    loader_out, load_err = _safe_call_loader(smoke_dir)
    if load_err is not None:
        return {**base_result, "ran": False, "ok": False, "errors": [load_err]}

    strategy_result = _build_predictions_for_strategy(
        loader_out=loader_out,
        interaction_model=interaction_model,
        np=np,
    )
    if strategy_result.get("error"):
        return {
            **base_result,
            "ran": False,
            "ok": False,
            "errors": [strategy_result["error"]],
        }

    predictions = strategy_result["predictions"]
    test_gt_for_direct = strategy_result["test_gt"]

    # Lead with the public path in diagnostics.
    saved_path = list(sys.path)
    sys.path.insert(0, str(smoke_dir))
    _purge_polluted_modules()
    modes: Dict[str, Any] = {}
    try:
        modes["agent_view"] = _run_mode_agent_view(
            smoke_dir=smoke_dir,
            interaction_model=interaction_model,
            predictions=predictions,
        )
        modes["direct"] = _run_mode_direct(
            smoke_dir=smoke_dir,
            predictions=predictions,
            test_gt=test_gt_for_direct,
        )
    finally:
        sys.path[:] = saved_path
        _purge_polluted_modules()

    errors: list[str] = []
    for mode_name, mode_result in modes.items():
        if not mode_result.get("ok"):
            err = mode_result.get("error") or "unknown failure"
            errors.append(f"{mode_name}:{err}")

    return {
        **base_result,
        "ran": True,
        "ok": not errors,
        "errors": errors,
        "modes": modes,
    }


def write_agent_smoke_failure_sidecar(
    staging_dir: Path,
    *,
    scope_id: str,
    task_id: str,
    result: dict,
) -> Path:
    """Write the latest smoke failure beside the staged evaluator."""
    payload = {
        "schema_version": 1,
        "phase_tag": "phase4:benchmark",
        "stage": "agent_smoke",
        "scope_id": scope_id,
        "task_id": task_id,
        "staging_dir": staging_dir.resolve().as_posix(),
        "validated_at_utc": datetime.now(tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "ran": bool(result.get("ran", False)),
        "ok": bool(result.get("ok", False)),
        "errors": list(result.get("errors", [])),
        "interaction_model": str(result.get("interaction_model", "")),
        "smoke_dir": str(result.get("smoke_dir", "")),
        "modes": _serialize_modes(result.get("modes", {})),
    }
    staging_dir.mkdir(parents=True, exist_ok=True)
    sidecar = staging_dir / _SMOKE_FAILURE_SIDECAR
    sidecar.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return sidecar


def format_smoke_failure_for_llm(result: dict) -> str:
    """Format bounded failure feedback for the next generation attempt."""
    if result.get("ok"):
        return ""
    lines: list[str] = ["Agent smoke gate failed."]
    modes = result.get("modes", {}) or {}
    # Preserve execution order in the feedback.
    for mode_name in ("agent_view", "direct"):
        mode = modes.get(mode_name)
        if not isinstance(mode, dict):
            continue
        if mode.get("ok"):
            continue
        err = mode.get("error") or "unknown failure"
        preamble = _human_preamble_for_mode(mode_name)
        lines.append("")
        lines.append(f"{preamble}: {_one_line(err)}")
        tb = mode.get("traceback") or ""
        if tb:
            lines.append("")
            lines.append(tb.rstrip())
    # Layout and loader failures occur before either mode runs.
    if len(lines) == 1:
        for err in result.get("errors", []) or []:
            lines.append(f"  {err}")
    text = "\n".join(lines)
    if len(text.encode("utf-8")) > _LLM_FEEDBACK_BYTE_BUDGET:
        encoded = text.encode("utf-8")[:_LLM_FEEDBACK_BYTE_BUDGET]
        text = encoded.decode("utf-8", errors="ignore")
        text += "\n\n[truncated for LLM feedback]"
    return text


def _materialize_smoke_layout(
    *,
    staging_dir: Path,
    data_dir: Path,
    smoke_dir: Path,
) -> list[str]:
    """Materialize an installed-style data directory; return copy errors."""
    errors: list[str] = []
    if smoke_dir.exists():
        shutil.rmtree(str(smoke_dir), ignore_errors=True)
    try:
        smoke_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        errors.append(f"smoke_dir_mkdir:{exc}")
        return errors

    for fname in ("task.py", "evaluator.py", "load.py", "task_rewards.py"):
        src = staging_dir / fname
        dst = smoke_dir / fname
        try:
            shutil.copy2(str(src), str(dst))
        except OSError as exc:
            errors.append(f"copy_{fname}:{exc}")
            return errors

    # Copy files rather than relying on platform-specific symlinks.
    try:
        for src in data_dir.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(data_dir)
            dst = smoke_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
    except OSError as exc:
        errors.append(f"copy_payload:{exc}")
        return errors

    return errors


def _safe_call_loader(smoke_dir: Path) -> tuple[dict | None, str | None]:
    """Import the bundled load.py and call ``load(smoke_dir)``."""
    load_unique = f"_smoke_loader_{abs(hash(str(smoke_dir / 'load.py')))}"
    try:
        loader_module = _load_module_from_path(load_unique, smoke_dir / "load.py")
    except Exception as exc:  # noqa: BLE001 — surface as gate error
        return None, f"load_module_import:{type(exc).__name__}:{exc}"
    try:
        loader_out = loader_module.load(smoke_dir)
    except Exception as exc:  # noqa: BLE001
        return None, f"load_call:{type(exc).__name__}:{exc}"
    finally:
        sys.modules.pop(load_unique, None)

    if not isinstance(loader_out, dict):
        return None, (
            "load_return_shape: load() must return a dict; "
            f"got {type(loader_out).__name__}"
        )
    return loader_out, None


def _build_predictions_for_strategy(
    *,
    loader_out: dict,
    interaction_model: str,
    np: Any,
) -> Dict[str, Any]:
    """Build deterministic, target-shaped predictions for contract testing."""
    if "reference" in loader_out:
        return _strategy_unconditional_generative(loader_out, np)
    if "test" in loader_out and isinstance(loader_out["test"], dict):
        return _strategy_b_shape(loader_out, interaction_model, np)
    return {
        "predictions": None,
        "test_gt": None,
        "error": (
            "load_return_shape: load() output must contain either a "
            "'reference' key (unconditional generative) or a 'test' "
            f"bundle (B-shape); got keys={sorted(loader_out.keys())}"
        ),
    }


def _strategy_unconditional_generative(
    loader_out: dict,
    np: Any,
) -> Dict[str, Any]:
    reference = loader_out.get("reference")
    if reference is None:
        return {
            "predictions": None,
            "test_gt": None,
            "error": "load_reference_none: 'reference' key is None",
        }
    try:
        # Echoing the reference tests metric plumbing, not model quality.
        ref_array = np.asarray(reference, dtype=np.float64)
        if ref_array.size == 0:
            return {
                "predictions": None,
                "test_gt": None,
                "error": "reference_empty: 'reference' has zero size",
            }
        predictions = ref_array.copy()
        return {
            "predictions": predictions,
            "test_gt": ref_array,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "predictions": None,
            "test_gt": None,
            "error": f"strategy_unconditional:{type(exc).__name__}:{exc}",
        }


def _strategy_b_shape(
    loader_out: dict,
    interaction_model: str,
    np: Any,
) -> Dict[str, Any]:
    test_bundle = loader_out["test"]
    train_bundle = loader_out.get("train") or {}
    test_gt = test_bundle.get("ground_truth")
    train_gt = train_bundle.get("ground_truth")
    if test_gt is None:
        return {
            "predictions": None,
            "test_gt": None,
            "error": (
                "load_test_ground_truth_none: B-shape loader returned "
                "test.ground_truth=None; the install slicer omits this "
                "from the agent dataset.h5 but the LLM loader's "
                "pre-install output must contain it."
            ),
        }
    if train_gt is None:
        return {
            "predictions": None,
            "test_gt": None,
            "error": (
                "load_train_ground_truth_none: B-shape loader returned "
                "train.ground_truth=None; train labels are required for "
                "the deterministic mean strategy."
            ),
        }

    try:
        # Flatten panel symbols using the same convention as dry-fire.
        test_gt_concat = _concat_panel_or_array(test_gt, np)
        train_gt_concat = _concat_panel_or_array(train_gt, np)
    except Exception as exc:  # noqa: BLE001
        return {
            "predictions": None,
            "test_gt": None,
            "error": f"strategy_concat:{type(exc).__name__}:{exc}",
        }

    if train_gt_concat.size == 0:
        return {
            "predictions": None,
            "test_gt": None,
            "error": "train_gt_empty: train ground_truth has zero size",
        }

    try:
        # Broadcast the training mean to the held-out target shape.
        if train_gt_concat.ndim == 1:
            mean_value = float(train_gt_concat.mean())
            predictions = np.full_like(
                np.asarray(test_gt_concat, dtype=np.float64),
                mean_value,
            )
        else:
            mean_vec = train_gt_concat.mean(axis=0)
            test_shape = np.asarray(test_gt_concat, dtype=np.float64).shape
            # Fall back to a scalar when trailing dimensions disagree.
            if (
                mean_vec.ndim >= 1
                and test_shape[-1:] == mean_vec.shape[-1:]
            ):
                predictions = np.broadcast_to(mean_vec, test_shape).copy()
            else:
                predictions = np.full(
                    test_shape, float(train_gt_concat.mean()), dtype=np.float64
                )
    except Exception as exc:  # noqa: BLE001
        return {
            "predictions": None,
            "test_gt": None,
            "error": f"strategy_broadcast:{type(exc).__name__}:{exc}",
        }

    _ = interaction_model
    return {
        "predictions": predictions,
        "test_gt": test_gt_concat,
        "error": None,
    }


def _concat_panel_or_array(value: Any, np: Any) -> Any:
    """Concatenate panel-of-symbols dict to a contiguous ndarray, or pass through."""
    if isinstance(value, dict):
        if not value:
            return np.empty((0,), dtype=np.float64)
        arrays = [np.asarray(v, dtype=np.float64) for v in value.values()]
        return np.concatenate(arrays, axis=0)
    return np.asarray(value, dtype=np.float64)


def _run_mode_agent_view(
    *,
    smoke_dir: Path,
    interaction_model: str,
    predictions: Any,
) -> Dict[str, Any]:
    """Mode B — instantiate ``task.py`` and call its agent-side scoring."""
    task_unique = f"_smoke_task_{abs(hash(str(smoke_dir / 'task.py')))}"
    captured = io.StringIO()
    try:
        try:
            from openfinai_pipeline.benchmark.contracts import (
                ForecastingTask,
                GenerativeTask,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            return _failure_mode_result(
                f"contracts_import:{type(exc).__name__}:{exc}",
                traceback.format_exc(limit=5),
            )

        try:
            task_module = _load_module_from_path(task_unique, smoke_dir / "task.py")
        except Exception as exc:  # noqa: BLE001
            return _failure_mode_result(
                f"task_module_import:{type(exc).__name__}:{exc}",
                traceback.format_exc(limit=5),
            )

        task_cls = _find_task_class(task_module, ForecastingTask, GenerativeTask)
        if task_cls is None:
            return _failure_mode_result(
                "task_class_not_found: no ForecastingTask/GenerativeTask "
                f"subclass exposed by {(smoke_dir / 'task.py').as_posix()}",
                None,
            )

        try:
            with contextlib.redirect_stdout(captured):
                task = task_cls(config={"data_dir": str(smoke_dir)})
        except Exception as exc:  # noqa: BLE001
            return _failure_mode_result(
                f"task_init:{type(exc).__name__}:{exc}",
                traceback.format_exc(limit=8),
            )

        # Distinguish a missing constructor from a swallowed evaluator import.
        if getattr(task, "_evaluator", None) is None:
            if "__init__" not in vars(task_cls):
                return _failure_mode_result(
                    "task_evaluator_unwired: task.py's class does not "
                    "define __init__, so the base-class default ran and "
                    "self._evaluator is None. Carry the template "
                    "reference's __init__ block through verbatim — that "
                    "is what wires the assembled evaluator.",
                    None,
                )
            return _failure_mode_result(
                "task_evaluator_unwired: task.__init__ ran but "
                "self._evaluator is None. The template's "
                "'from evaluator import ...' is wrapped in try/except, "
                "so a wrong class name (or any other ImportError) "
                "silently leaves the field unset. Check that the "
                "imported class name matches the evaluator module's "
                "exported class.",
                None,
            )

        # The agent path may score only the training split.
        try:
            with contextlib.redirect_stdout(captured):
                if isinstance(task, ForecastingTask):
                    scores = task.predict_and_evaluate(predictions, split="train")
                elif isinstance(task, GenerativeTask):
                    scores = task.generate_and_evaluate(predictions, split="train")
                else:
                    return _failure_mode_result(
                        f"unsupported_task_baseclass: {type(task).__mro__}; "
                        f"expected ForecastingTask or GenerativeTask "
                        f"(interaction_model={interaction_model!r})",
                        None,
                    )
        except Exception as exc:  # noqa: BLE001
            return _failure_mode_result(
                f"agent_view_score:{type(exc).__name__}:{exc}",
                traceback.format_exc(limit=8),
            )

        return _evaluate_score_result(
            scores=scores,
            mode_name="agent_view",
            captured_stdout=captured.getvalue(),
        )
    finally:
        sys.modules.pop(task_unique, None)
        text = captured.getvalue().strip()
        if text:
            log_detail(logger, "agent_smoke agent_view stdout:\n%s", text)


def _run_mode_direct(
    *,
    smoke_dir: Path,
    predictions: Any,
    test_gt: Any,
) -> Dict[str, Any]:
    """Mode A — load assembled evaluator and call ``score()`` directly."""
    eval_unique = f"_smoke_evaluator_{abs(hash(str(smoke_dir / 'evaluator.py')))}"
    captured = io.StringIO()
    try:
        try:
            evaluator_module = _load_module_from_path(
                eval_unique, smoke_dir / "evaluator.py"
            )
        except Exception as exc:  # noqa: BLE001
            return _failure_mode_result(
                f"evaluator_module_import:{type(exc).__name__}:{exc}",
                traceback.format_exc(limit=5),
            )

        evaluator_cls = _find_evaluator_class(evaluator_module)
        if evaluator_cls is None:
            return _failure_mode_result(
                "evaluator_class_not_found: no BaseEvaluator subclass exposed "
                f"by {(smoke_dir / 'evaluator.py').as_posix()}",
                None,
            )

        try:
            evaluator = evaluator_cls()
        except Exception as exc:  # noqa: BLE001
            return _failure_mode_result(
                f"evaluator_init:{type(exc).__name__}:{exc}",
                traceback.format_exc(limit=5),
            )

        try:
            with contextlib.redirect_stdout(captured):
                scores = evaluator.score(
                    predictions=predictions,
                    ground_truth=test_gt,
                    data_dir=smoke_dir,
                )
        except Exception as exc:  # noqa: BLE001
            return _failure_mode_result(
                f"direct_score:{type(exc).__name__}:{exc}",
                traceback.format_exc(limit=8),
            )

        return _evaluate_score_result(
            scores=scores,
            mode_name="direct",
            captured_stdout=captured.getvalue(),
        )
    finally:
        sys.modules.pop(eval_unique, None)
        text = captured.getvalue().strip()
        if text:
            log_detail(logger, "agent_smoke direct stdout:\n%s", text)


def _evaluate_score_result(
    *,
    scores: Any,
    mode_name: str,
    captured_stdout: str,
) -> Dict[str, Any]:
    """Translate a ``score()`` return value into a mode-result dict."""
    if not isinstance(scores, dict):
        return _failure_mode_result(
            f"{mode_name}_score_return_type: expected dict, "
            f"got {type(scores).__name__}",
            None,
        )

    # A global evaluator error always fails the mode.
    global_err = scores.get("_error")
    if global_err:
        return {
            "ok": False,
            "scores": _to_jsonable(scores),
            "error": f"{mode_name}_score_global_error:{global_err}",
            "traceback": None,
        }

    # Per-metric errors are allowed when the aggregate reward remains finite.
    reward_total = scores.get("reward")
    if reward_total is None:
        return {
            "ok": False,
            "scores": _to_jsonable(scores),
            "error": f"{mode_name}_missing_reward_aggregator",
            "traceback": None,
        }
    try:
        import math

        if not math.isfinite(float(reward_total)):
            return {
                "ok": False,
                "scores": _to_jsonable(scores),
                "error": (
                    f"{mode_name}_reward_non_finite: reward={reward_total!r}"
                ),
                "traceback": None,
            }
    except (TypeError, ValueError) as exc:
        return {
            "ok": False,
            "scores": _to_jsonable(scores),
            "error": (
                f"{mode_name}_reward_non_numeric:{type(exc).__name__}:{exc}"
            ),
            "traceback": None,
        }

    return {
        "ok": True,
        "scores": _to_jsonable(scores),
        "error": None,
        "traceback": None,
    }


def _failure_mode_result(error: str, tb: str | None) -> Dict[str, Any]:
    return {
        "ok": False,
        "scores": {},
        "error": error,
        "traceback": tb,
    }


def _purge_polluted_modules() -> None:
    """Drop bare-name and hash-suffixed modules introduced by the smoke."""
    for name in _BARE_NAME_MODULES_TO_CLEAN:
        sys.modules.pop(name, None)
    for k in list(sys.modules):
        if k.startswith(_HASHED_PREFIXES_TO_CLEAN):
            sys.modules.pop(k, None)


def _load_module_from_path(module_name: str, module_path: Path) -> ModuleType:
    """Load a staged module under an isolated name."""
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_task_class(
    module: ModuleType,
    forecasting_base: type,
    generative_base: type,
) -> type | None:
    """Find the generated forecasting or generative task class."""
    candidates: list[type] = []
    for attr in dir(module):
        if attr.startswith("_"):
            continue
        obj = getattr(module, attr, None)
        if not isinstance(obj, type):
            continue
        if obj is forecasting_base or obj is generative_base:
            continue
        if obj.__module__ != module.__name__:
            continue
        if issubclass(obj, forecasting_base) or issubclass(obj, generative_base):
            candidates.append(obj)
    return candidates[0] if candidates else None


def _human_preamble_for_mode(mode_name: str) -> str:
    if mode_name == "agent_view":
        return (
            "Mode AGENT-VIEW failed (instantiate task.py + call "
            "predict_and_evaluate / generate_and_evaluate, split='train')"
        )
    if mode_name == "direct":
        return (
            "Mode DIRECT failed (call evaluator.score(predictions, "
            "ground_truth) without going through task.py)"
        )
    return f"Mode {mode_name.upper()} failed"


def _one_line(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) > 400:
        cleaned = cleaned[:400] + "..."
    return cleaned


def _serialize_modes(modes: Any) -> dict:
    out: dict[str, Any] = {}
    if not isinstance(modes, dict):
        return out
    for k, v in modes.items():
        if not isinstance(v, dict):
            continue
        out[k] = {
            "ok": bool(v.get("ok", False)),
            "error": v.get("error"),
            "traceback": v.get("traceback"),
            "scores": _to_jsonable(v.get("scores", {})),
        }
    return out


def _to_jsonable(value: Any) -> Any:
    """Best-effort convert a score dict to JSON-serialisable scalars."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
