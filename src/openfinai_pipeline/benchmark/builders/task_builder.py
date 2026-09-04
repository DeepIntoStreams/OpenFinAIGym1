"""Build and validate Harbor task packages from accepted paper artifacts.

The LLM authors task.py and short prose slots. The pipeline assembles the
instruction, evaluator, loader, contract tests, and manifest.
"""

import ast
import json
import logging
import re
import shutil
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from openfinai_pipeline.corpus.aggregator import paper_global_sort_key
from openfinai_pipeline.corpus.catalog import (
    slugify,
)
from openfinai_pipeline.llm import LLMService
from openfinai_pipeline.papers.pdf_downloader import PDFDownloader
from openfinai_pipeline.papers.summary_payload import extract_summary_payload
from openfinai_pipeline.prompts import (
    build_fix_errors_prompt,
    build_implementation_prompt,
    build_task_dedup_prompt,
    task_dedup_schema,
    task_fix_schema,
    task_generation_schema,
)
from openfinai_pipeline.benchmark.agent_smoke import (
    agent_smoke_validate,
    format_smoke_failure_for_llm,
    write_agent_smoke_failure_sidecar,
)
from openfinai_pipeline.settings import DownloadConfig
from openfinai_pipeline.summary import strip_low_value_sections
from openfinai_pipeline.utils.logging import fmt_chars, log_detail, log_stage, truncate_oneline
from openfinai_pipeline.utils.sandbox import Sandbox

logger = logging.getLogger(__name__)
PHASE_TAG = "[phase4:benchmark]"

# Shared prompt cap, approximating four characters per token.
_PAPER_CONTEXT_MAX_CHARS = 60_000  # ~15k tokens
_INIT_PY_DEFAULT = ""
_EXPLICIT_RUNNER_NAME = "run_evaluation_explicit.py"
_HARBOR_SUBMIT_NAME = "harbor_submit.py"

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

_INLINE_EXAMPLE_TASK = '''\
"""Minimal example task for reference.

Note for the LLM author: ``load_data`` is inherited from ``BaseTask``
and imports the bundled per-task ``load.py`` deterministically. Do NOT
override it. Implement only the abstract methods specific to this
task's interaction model.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from openfinai_pipeline.benchmark.contracts import BaseTask, TaskMetadata

class ExampleTask(BaseTask):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._current_step = 0
        self._done = False
        self._returns: List[float] = []

    def metadata(self) -> TaskMetadata:
        return TaskMetadata(
            task_id="example", title="Example", description="Minimal example.",
            data_requirements=["synthetic data"], version="1.0.0",
        )

    # load_data inherited — returns self._data populated by load.py.
    # self._data["features"] and self._data["ground_truth"] are available
    # after the first call.

    def get_observation_space(self) -> Dict[str, Any]:
        return {"shape": (5,), "dtype": "float32"}

    def get_action_space(self) -> Dict[str, Any]:
        return {"type": "continuous", "shape": (5,)}

    def reset(self) -> Any:
        self._current_step = 0
        self._done = False
        self._returns = []
        if self._data is None:
            self.load_data()
        return self._data["features"][0]

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        feats = self._data["features"]
        self._current_step += 1
        done = self._current_step >= len(feats) - 1
        self._done = done
        reward = float(np.dot(action, feats[self._current_step]))
        self._returns.append(reward)
        obs = feats[min(self._current_step, len(feats) - 1)]
        return obs, reward, done, {}

    @staticmethod
    def _calculate_rewards(returns: List[float], trades: List[Dict]) -> Dict[str, float]:
        rewards: Dict[str, float] = {"total_return": float(np.sum(returns)), "num_trades": len(trades)}
        if returns:
            arr = np.array(returns)
            mean_r, std_r = float(np.mean(arr)), float(np.std(arr))
            rewards["mean_return"] = mean_r
            rewards["std_return"] = std_r
            rewards["sharpe_ratio"] = mean_r / std_r if std_r != 0 else 0.0
            downside = arr[arr < 0]
            ds_std = float(np.std(downside)) if len(downside) > 0 else 0.0
            rewards["sortino_ratio"] = mean_r / ds_std if ds_std != 0 else (float("inf") if mean_r > 0 else 0.0)
            cumulative = np.cumprod(1 + arr)
            running_max = np.maximum.accumulate(cumulative)
            rewards["max_drawdown"] = float(np.min((cumulative - running_max) / running_max))
        if trades:
            rewards["win_rate"] = len([t for t in trades if t.get("profit", 0) > 0]) / len(trades)
        return rewards

    def evaluate(self, agent_actions: List[Any], **kw) -> Dict[str, float]:
        self.reset()
        for a in agent_actions:
            if self._done:
                break
            _, r, _, _ = self.step(a)
        return self._calculate_rewards(self._returns, [])
'''


_INLINE_EXAMPLE_FORECASTING_TASK = '''\
"""Minimal forecasting task example.

Note for the LLM author: ``load_data`` is inherited from
``ForecastingTask`` (via ``BaseTask``) and imports the bundled per-task
``load.py`` deterministically. After the first call,
``self._data["features"]`` and ``self._data["ground_truth"]`` are
populated by Phase 4's loader. Do NOT override ``load_data`` and do
NOT add any data parsing inside ``task.py``.
"""
from typing import Any, Dict, Optional
from openfinai_pipeline.benchmark.contracts import ForecastingTask, TaskMetadata


class ExampleForecastingTask(ForecastingTask):
    """Predicts next-day returns from packaged features.

    The agent receives the full feature matrix at once and returns all
    predictions.  Scoring is delegated to the sibling evaluator module via
    ForecastingTask.predict_and_evaluate -- no reset/step/evaluate override
    is needed.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        # Wire the assembled evaluator if available in this package.
        try:
            from evaluator import ExampleForecastingTaskEvaluator  # type: ignore
            self._evaluator = ExampleForecastingTaskEvaluator()
        except Exception:
            self._evaluator = None

    def metadata(self) -> TaskMetadata:
        return TaskMetadata(
            task_id="example_forecasting",
            title="Example Forecasting",
            description="Predict next-period returns from historical features.",
            interaction_model="forecasting",
            data_requirements=["synthetic"],
            version="1.0.0",
        )

    # load_data is INHERITED -- it imports the bundled load.py and sets
    # self._data to {"features": ndarray, "ground_truth": ndarray}.

    def get_features(self) -> Any:
        self.load_data()
        return self._data["features"]

    def get_ground_truth(self) -> Any:
        self.load_data()
        return self._data["ground_truth"]

    def get_observation_space(self) -> Dict[str, Any]:
        return {"shape": (5,), "dtype": "float32"}

    def get_action_space(self) -> Dict[str, Any]:
        return {"type": "continuous", "shape": (1,)}
'''


_INLINE_EXAMPLE_GENERATIVE_TASK = '''\
"""Minimal generative task example.

Note for the LLM author: ``load_data`` is inherited from
``GenerativeTask`` (via ``BaseTask``) and imports the bundled per-task
``load.py`` deterministically. The loader populates
``self._data["ground_truth"]`` with the reference samples (canonical
scoring key) and may populate ``self._data["features"]`` with
conditioning context. Do NOT override ``load_data``.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from openfinai_pipeline.benchmark.contracts import GenerativeTask, TaskMetadata


class ExampleGenerativeTask(GenerativeTask):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        try:
            from evaluator import ExampleGenerativeTaskEvaluator  # type: ignore
            self._evaluator = ExampleGenerativeTaskEvaluator()
        except Exception:
            self._evaluator = None

    def metadata(self) -> TaskMetadata:
        return TaskMetadata(
            task_id="example_generative",
            title="Example Generative",
            description="Generate samples that match a reference distribution.",
            interaction_model="generative",
            data_requirements=["synthetic"],
            version="1.0.0",
        )

    # load_data is INHERITED -- it imports the bundled load.py and sets
    # self._data to {"features": optional ndarray (conditioning),
    # "ground_truth": ndarray (reference samples)}.

    def get_reference_data(self) -> Any:
        self.load_data()
        return self._data["ground_truth"]

    def get_observation_space(self) -> Dict[str, Any]:
        return {"shape": (24, 3), "dtype": "float32"}

    def get_action_space(self) -> Dict[str, Any]:
        return {"type": "continuous", "shape": (24, 3)}
'''


def _normalize_task_family(task_family: str) -> str:
    value = str(task_family or "").strip().lower()
    if value in {
        "forecasting",
        "generative",
        "trading",
        "realtime_forecasting",
        "realtime_trading",
    }:
        return value
    return ""


def _infer_interaction_model(candidate: "TaskCandidate") -> str:
    """Return the interaction model used for implementation routing."""
    task_family = _normalize_task_family(candidate.task_family)
    if task_family == "realtime_forecasting":
        return "forecasting"
    if task_family == "realtime_trading":
        return "trading"
    if task_family in {"forecasting", "generative", "trading"}:
        return task_family

    # Fallback only when task_family is missing.
    text = ""
    try:
        text = (candidate.ml_task_summary or "") + " " + (candidate.title or "")
    except Exception:
        return "gym"
    text = text.lower()

    forecasting_keywords = (
        "predict", "forecast", "classification", "regression",
        "detection", "estimation", "sentiment", "forecasting",
        "prediction", "movement prediction",
    )
    trading_keywords = (
        "trading", "portfolio", "execution", "allocation",
        "rebalanc", "order book", "market making", "pairs trading",
    )
    fc_score = sum(1 for kw in forecasting_keywords if kw in text)
    tr_score = sum(1 for kw in trading_keywords if kw in text)
    if fc_score > tr_score:
        return "forecasting"
    if tr_score > fc_score:
        return "trading"
    return "gym"


# Data classes


@dataclass
class TaskCandidate:
    """A harvested task ready for implementation."""

    task_dir: str
    task_id: str
    title: str
    scope_id: str
    paper_id: str
    ml_task_summary: str
    experiments: str
    datasets: list[dict[str, str]]
    rewards: list[dict[str, Any]]
    paper_abstract: str
    pdf_path: str | None
    task_family: str = ""


@dataclass
class GeneratedCode:
    """Generated task source plus prose and pipeline-rendered artifacts."""

    task_py: str
    context_prose: str = ""
    objective_prose: str = ""
    schema_description: str = ""
    instruction_md: str = ""
    evaluator_py: str = ""
    init_py: str = _INIT_PY_DEFAULT


@dataclass
class ImplementationResult:
    """Outcome of a single task implementation attempt."""

    task_id: str
    success: bool
    module_path: str | None = None
    attempts: int = 0
    error_log: list[str] = field(default_factory=list)
    # Final failed gate, or None before validation and on success.
    failure_stage: str | None = None


@dataclass
class DatasetArtifact:
    """Phase 2 dataset artifact enrichment data."""

    name: str
    asset_dir_name: str
    downloaded_dataset_description: str
    download_script: str  # full source code of the approved download script
    approved: bool
    payload_files: list[str] = field(default_factory=list)
    source_paths: list[str] = field(default_factory=list)
    preview: list[dict[str, Any]] = field(default_factory=list)
    data_dir_name: str = ""
    source_kind: str = "paper_summary"
    # Manifest roles replace filename heuristics during target loading.
    interaction_model: str = ""
    has_ground_truth: bool = False
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    manifest_status: str = ""


@dataclass
class RewardArtifact:
    """Reward enrichment data from one of three sources."""

    name: str
    source: str  # "phase3_reward_fn" | "torch_library" | "utils" | "unmatched"
    paper_metric_description: str
    code: str  # reward_fn source OR torch class source
    approved: bool
    module_name: str = ""
    class_name: str = ""
    base_class: str = ""  # "Loss" | "TradingReward" — drives runtime dispatch


@dataclass
class PhaseArtifacts:
    """Collected Phase 2/3 artifacts for a single task candidate."""

    datasets: list[DatasetArtifact] = field(default_factory=list)
    rewards: list[RewardArtifact] = field(default_factory=list)
    unmatched_rewards: list[dict[str, Any]] = field(default_factory=list)


# Phase 2/3 enrichment helpers


def _slugify(name: str) -> str:
    """Consistent slug generation (mirrors corpus catalog slugify)."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return clean or "reward"


def _read_latest_script(directory: Path, prefix: str, ext: str = ".py") -> str:
    """Read the latest numbered script (e.g. download3.py -> download2.py -> download1.py)."""
    for i in range(3, 0, -1):
        path = directory / f"{prefix}{i}{ext}"
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                continue
    return ""


def _extract_torch_class_source(rewards_py_path: Path, class_name: str) -> str:
    """Extract a single class definition from the torch reward_bank.py using AST."""
    try:
        source = rewards_py_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return ""

    source_lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            start = node.lineno - 1
            end = (
                node.end_lineno
                if hasattr(node, "end_lineno") and node.end_lineno
                else start + 1
            )
            return "".join(source_lines[start:end])
    return ""


_CATALOG_CACHE: dict[str, list[dict[str, Any]]] = {}


def _load_catalog(reward_bank_json_path: Path) -> list[dict[str, Any]]:
    """Load ``reward_bank.json`` once per process; cached by absolute path."""
    key = str(reward_bank_json_path.resolve())
    if key in _CATALOG_CACHE:
        return _CATALOG_CACHE[key]
    try:
        data = json.loads(reward_bank_json_path.read_text(encoding="utf-8"))
        entries = data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        entries = []
    _CATALOG_CACHE[key] = entries
    return entries


def _catalog_entry_for_class(
    class_name: str, reward_bank_json_path: Path
) -> dict[str, Any] | None:
    """Look up a ``reward_bank.json`` entry by ``class_name``."""
    if not class_name:
        return None
    for entry in _load_catalog(reward_bank_json_path):
        if str(entry.get("class_name", "")).strip() == class_name:
            return entry
    return None


def _load_generated_spec(
    script_path: Path, *, class_name: str = ""
) -> dict[str, Any] | None:
    """Load a generated reward spec by class, module, then legacy sidecar."""
    catalog_path = script_path.parent / "generated_rewards.json"
    if catalog_path.is_file():
        try:
            entries = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = []
        if isinstance(entries, list):
            module_stem = script_path.stem
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entry_class = str(entry.get("class_name", "")).strip()
                entry_module = str(entry.get("module", "")).strip()
                if class_name and entry_class == class_name:
                    return entry
                if entry_module and entry_module == module_stem:
                    return entry
    legacy_path = script_path.with_suffix(".spec.json")
    if legacy_path.is_file():
        try:
            return json.loads(legacy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _ast_class_method_params(
    source_code: str, class_name: str
) -> tuple[set[str], set[str]]:
    """Return (init_param_names, forward-or-compute param_names) for a class.

    Uses AST so no import-time side effects; tolerates parse errors by
    returning empty sets. ``self`` and any ``*args``/``**kwargs`` are excluded.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return set(), set()

    def _params(fn: ast.FunctionDef) -> set[str]:
        names: set[str] = set()
        for a in fn.args.args:
            if a.arg != "self":
                names.add(a.arg)
        for a in fn.args.kwonlyargs:
            names.add(a.arg)
        return names

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        init_params: set[str] = set()
        forward_params: set[str] = set()
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if item.name == "__init__":
                init_params = _params(item)
            elif item.name == "forward" and not forward_params:
                forward_params = _params(item)
            elif item.name == "compute" and not forward_params:
                forward_params = _params(item)
            elif item.name == "compute_aggregate":
                # TradingReward case — takes precedence if present.
                forward_params = _params(item)
        return init_params, forward_params
    return set(), set()


def _ast_required_init_param_names(
    source_code: str, class_name: str
) -> set[str]:
    """Return required named ``__init__`` parameters, excluding ``self``."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if not (isinstance(item, ast.FunctionDef) and item.name == "__init__"):
                continue
            pos_args = list(item.args.posonlyargs) + list(item.args.args)
            n_defaults = len(item.args.defaults)
            n_pos = len(pos_args)
            required: set[str] = set()
            for idx, arg in enumerate(pos_args):
                if arg.arg == "self":
                    continue
                has_default = idx >= (n_pos - n_defaults)
                if not has_default:
                    required.add(arg.arg)
            for arg, kw_default in zip(item.args.kwonlyargs, item.args.kw_defaults):
                if kw_default is None:
                    required.add(arg.arg)
            return required
    return set()


def _format_signature_from_ast(fn: ast.FunctionDef) -> str:
    """Render an unannotated argument signature from a function AST."""
    pos_args = list(fn.args.posonlyargs) + list(fn.args.args)
    defaults = list(fn.args.defaults)
    n_pos_no_default = len(pos_args) - len(defaults)
    parts: list[str] = []
    for idx, arg in enumerate(pos_args):
        if arg.arg == "self":
            continue
        if idx < n_pos_no_default:
            parts.append(arg.arg)
        else:
            default = defaults[idx - n_pos_no_default]
            parts.append(f"{arg.arg}={ast.unparse(default)}")
    if fn.args.vararg is not None:
        parts.append(f"*{fn.args.vararg.arg}")
    elif fn.args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        if default is None:
            parts.append(arg.arg)
        else:
            parts.append(f"{arg.arg}={ast.unparse(default)}")
    if fn.args.kwarg is not None:
        parts.append(f"**{fn.args.kwarg.arg}")
    return f"({', '.join(parts)})"


def _ast_class_summary(
    source_code: str, class_name: str
) -> dict[str, str]:
    """Extract a reward's docstring, signatures, and first base class."""
    blank = {
        "docstring": "",
        "init_signature": "",
        "forward_signature": "",
        "base_class": "",
    }
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return blank
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        out = dict(blank)
        out["docstring"] = (ast.get_docstring(node) or "").strip()
        # Prefer a recognized reward base; otherwise keep the first base name.
        for base in node.bases:
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if not base_name:
                continue
            if base_name in ("Loss", "TradingReward"):
                out["base_class"] = base_name
                break
            if not out["base_class"]:
                out["base_class"] = base_name
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if item.name == "__init__":
                out["init_signature"] = _format_signature_from_ast(item)
            elif item.name in ("forward", "compute_aggregate", "compute"):
                if not out["forward_signature"]:
                    out["forward_signature"] = _format_signature_from_ast(item)
        return out
    return blank


def build_metric_specs_for_instruction(
    *,
    resolved_reward_names: list[str],
    task_rewards_path: Path | None,
) -> list[dict[str, str]]:
    """Build ordered instruction blocks for resolved metrics.

    Curated semantics supply prediction contracts; bundled source supplies
    signatures and docstrings. Every resolved metric remains visible.
    """
    from data.knowledge_base.rewards.prediction_semantics import (
        FORECASTING_LOSS_SEMANTICS,
        GENERATIVE_LOSS_SEMANTICS,
        TRADING_REWARD_SEMANTICS,
        get_prediction_semantics,
    )

    bundled_source = ""
    if task_rewards_path is not None:
        try:
            bundled_source = Path(task_rewards_path).read_text(encoding="utf-8")
        except OSError:
            bundled_source = ""

    def _classify_base_from_curated(name: str) -> str:
        if name in TRADING_REWARD_SEMANTICS:
            return "TradingReward"
        if name in FORECASTING_LOSS_SEMANTICS or name in GENERATIVE_LOSS_SEMANTICS:
            return "Loss"
        return ""

    blank_summary = {
        "docstring": "",
        "init_signature": "",
        "forward_signature": "",
        "base_class": "",
    }
    specs: list[dict[str, str]] = []
    for cls_name in resolved_reward_names:
        cls_name = (cls_name or "").strip()
        if not cls_name:
            continue
        ast_summary = (
            _ast_class_summary(bundled_source, cls_name)
            if bundled_source
            else dict(blank_summary)
        )
        contract = get_prediction_semantics(cls_name)
        # Prefer curated classification, then the bundled class's first base.
        base_class = _classify_base_from_curated(cls_name) or ast_summary["base_class"]
        # The renderer handles metrics with no prose using its source fallback.
        specs.append(
            {
                "name": cls_name,
                "base_class": base_class,
                "init_signature": ast_summary["init_signature"],
                "forward_signature": ast_summary["forward_signature"],
                "docstring": ast_summary["docstring"],
                "prediction_contract": contract,
            }
        )
    return specs


def _resolve_reward_reference(
    reward_name: str,
    *,
    reward_description: str,
    reward_script_path: str,
    generated_root: Path,
    reward_bank_json_path: Path | None = None,
    resolved_class_name: str = "",
) -> dict[str, Any] | None:
    """Bind a reward to an exact class in its referenced script.

    Phase 3's resolved class takes precedence over the paper-side name. No
    fuzzy matching is used; missing sources or exact matches return ``None``.
    """
    script_path_raw = str(reward_script_path).strip()
    if not script_path_raw:
        return None
    script_path = Path(script_path_raw)
    if not script_path.is_file():
        return None

    resolved_script_path = script_path.resolve()
    try:
        source_code = resolved_script_path.read_text(encoding="utf-8")
    except OSError:
        return None

    class_candidates = _find_reward_class_candidates(source_code)
    if not class_candidates:
        return None

    # Fall back to the paper-side class name for legacy callers.
    target_name = str(resolved_class_name or "").strip() or str(reward_name).strip()
    if not target_name:
        return None
    selected = next(
        (c for c in class_candidates if c.get("class_name", "") == target_name),
        None,
    )
    if selected is None:
        return None

    module_name = resolved_script_path.stem
    is_generated = generated_root.resolve() in resolved_script_path.parents
    source = (
        "generated_reward_script_path"
        if is_generated
        else "curated_reward_script_path"
    )
    class_name = str(selected.get("class_name", "")).strip()
    class_source = str(selected.get("source", ""))

    # Only literal hyperparameter defaults require catalog metadata.
    spec_fields: dict[str, Any] = {}
    if is_generated:
        generated_spec = (
            _load_generated_spec(resolved_script_path, class_name=class_name) or {}
        )
        if "default_params" in generated_spec:
            spec_fields["default_params"] = generated_spec["default_params"]
    elif reward_bank_json_path is not None:
        catalog_entry = _catalog_entry_for_class(class_name, reward_bank_json_path)
        if catalog_entry is not None and "default_params" in catalog_entry:
            spec_fields["default_params"] = catalog_entry["default_params"]

    base_class = str(selected.get("base_class", "")).strip() or "Loss"
    spec_fields.setdefault("default_params", {})

    return {
        "source": source,
        "module_name": module_name,
        "class_name": class_name,
        "code": class_source,
        "base_class": base_class,
        "default_params": dict(spec_fields.get("default_params", {})),
    }


def _find_reward_class_candidates(source_code: str) -> list[dict[str, str]]:
    """Return top-level ``Loss`` subclasses eligible for auto-construction.

    Trading and event rewards have incompatible evaluator call shapes.
    """
    try:
        tree = ast.parse(source_code)
    except Exception:
        return []
    lines = source_code.splitlines(keepends=True)
    candidates: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names: list[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)
        if "Loss" not in base_names:
            continue
        start = node.lineno - 1
        end = node.end_lineno if getattr(node, "end_lineno", None) else start + 1
        candidates.append(
            {
                "class_name": node.name,
                "base_class": "Loss",
                "source": "".join(lines[start:end]),
            }
        )
    return candidates


def _load_dataset_artifact_from_summary_entry(
    dataset_entry: dict[str, Any],
    *,
    datasets_root: Path,
    curated_datasets_root: Path,
) -> DatasetArtifact | None:
    if not bool(dataset_entry.get("download_successful", False)):
        return None
    source_paths = [
        str(path).strip()
        for path in dataset_entry.get("dataset_paths", [])
        if str(path).strip()
    ]
    if not source_paths:
        return None
    dataset_name = str(dataset_entry.get("name", "")).strip()
    script_path = str(dataset_entry.get("download_script_path", "")).strip()
    data_root = Path(script_path).resolve().parent / "data" if script_path else None
    payload_files: list[str] = []
    datasets_root_resolved = datasets_root.resolve()
    curated_root_resolved = curated_datasets_root.resolve()
    for raw_path in source_paths:
        path_obj = Path(raw_path)
        resolved_path = path_obj.resolve()
        if data_root is not None:
            try:
                payload_files.append(resolved_path.relative_to(data_root).as_posix())
                continue
            except ValueError:
                pass
        try:
            payload_files.append(
                resolved_path.relative_to(curated_root_resolved).as_posix()
            )
            continue
        except ValueError:
            pass
        try:
            payload_files.append(
                resolved_path.relative_to(datasets_root_resolved).as_posix()
            )
            continue
        except ValueError:
            pass
        payload_files.append(path_obj.name)
    download_script = ""
    asset_dir_name = ""
    data_dir_name = "data"
    if script_path:
        try:
            download_script = Path(script_path).read_text(encoding="utf-8")
        except OSError:
            download_script = ""
        script_parent = Path(script_path).resolve().parent
        asset_dir_name = script_parent.name
    artifacts_entries = dataset_entry.get("artifacts", [])
    if not isinstance(artifacts_entries, list):
        artifacts_entries = []
    return DatasetArtifact(
        name=dataset_name,
        asset_dir_name=asset_dir_name or dataset_name,
        downloaded_dataset_description=str(
            dataset_entry.get("downloaded_dataset_description", "")
        ).strip()
        or str(dataset_entry.get("description", "")).strip(),
        download_script=download_script,
        approved=True,
        payload_files=payload_files,
        source_paths=source_paths,
        preview=_normalize_dataset_preview(dataset_entry.get("preview", [])),
        data_dir_name=data_dir_name,
        source_kind="paper_summary",
        interaction_model=str(dataset_entry.get("interaction_model", "")).strip(),
        has_ground_truth=bool(dataset_entry.get("has_ground_truth", False)),
        artifacts=[item for item in artifacts_entries if isinstance(item, dict)],
        manifest_status=str(dataset_entry.get("manifest_status", "")).strip(),
    )


def _normalize_dataset_preview(
    preview: Any,
    *,
    prefix: str = "",
) -> list[dict[str, Any]]:
    if isinstance(preview, dict):
        preview_items = preview.get("files", [])
    elif isinstance(preview, list):
        preview_items = preview
    else:
        preview_items = []
    normalized: list[dict[str, Any]] = []
    for item in preview_items:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        rel_path = str(entry.get("path", "")).strip()
        if prefix and rel_path and not rel_path.startswith(prefix):
            entry["path"] = f"{prefix}/{rel_path}"
        normalized.append(entry)
    return normalized


def enrich_candidate_from_phases(
    candidate: TaskCandidate,
    datasets_root: Path,
    curated_datasets_root: Path,
    rewards_root: Path,
    generated_dir: Path | None = None,
) -> PhaseArtifacts:
    """
    Look up matching Phase 2 datasets (via task_dirs) and Phase 3/torch/utils
    rewards for a task candidate. Returns enrichment artifacts to inject into
    the LLM prompt.
    """
    artifacts = PhaseArtifacts()

    # Fan-out should leave one dataset; tolerate malformed multi-dataset input.
    for dataset_entry in candidate.datasets:
        dataset_name = str(dataset_entry.get("name", "")).strip()
        if not dataset_name:
            continue
        selected = _load_dataset_artifact_from_summary_entry(
            dataset_entry,
            datasets_root=datasets_root,
            curated_datasets_root=curated_datasets_root,
        )
        if selected is not None:
            artifacts.datasets.append(selected)
            break

    # Resolve paper-local reward script paths.
    generated_root = generated_dir if generated_dir is not None else (rewards_root / "generated")

    for reward_entry in candidate.rewards:
        reward_name = reward_entry.get("name", "")
        reward_desc = reward_entry.get("description", "")
        if not reward_name:
            continue

        resolution = _resolve_reward_reference(
            reward_name,
            reward_description=reward_desc,
            reward_script_path=str(reward_entry.get("reward_script_path", "")).strip(),
            generated_root=generated_root,
            reward_bank_json_path=rewards_root / "reward_bank.json",
            resolved_class_name=str(
                reward_entry.get("resolved_class_name", "")
            ).strip(),
        )
        if resolution is not None:
            source = str(
                resolution.get("source", "generated_reward")
            ).strip() or "generated_reward"
            module_name = str(resolution.get("module_name", "")).strip()
            class_name = str(resolution.get("class_name", "")).strip()
            # Candidate discovery limits this branch to Loss subclasses.
            base_class = str(resolution.get("base_class", "")).strip() or "Loss"
            code = str(resolution.get("code", ""))
            artifacts.rewards.append(
                RewardArtifact(
                    name=reward_name,
                    source=source,
                    paper_metric_description=reward_desc,
                    code=code,
                    approved=bool(code.strip()),
                    module_name=module_name,
                    class_name=class_name,
                    base_class=base_class,
                )
            )
            if code.strip():
                continue

        artifacts.unmatched_rewards.append(
            {
                "name": reward_name,
                "description": reward_desc,
            }
        )

    logger.info(
        "%s   enrichment: %d datasets, %d rewards matched, %d unmatched",
        PHASE_TAG,
        len(artifacts.datasets),
        len(artifacts.rewards),
        len(artifacts.unmatched_rewards),
    )
    return artifacts


# Evaluator deterministic assembly (Phase 3)


def _staging_folder_name(task_id: str, ml_task_summary: str) -> str:
    """Derive a descriptive staging folder name from task_id + summary.

    Example: task1 + "The paper introduces XGBoost..." -> "task1_the_paper_introduces_xgboost"
    """
    words = re.sub(r"[^a-zA-Z0-9\\s]", "", ml_task_summary).split()[:6]
    slug = "_".join(w.lower() for w in words)[:50]
    return f"{task_id}_{slug}" if slug else task_id


def _find_subclass_name(source_code: str, *base_classes: str) -> str | None:
    try:
        tree = ast.parse(source_code)
    except Exception:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name in base_classes:
                    return node.name
    return None


def _detect_reward_call_pattern(source_code: str) -> str:
    """Infer how a reward class should be invoked."""
    class_name = _find_subclass_name(source_code, "TradingReward")
    if class_name:
        return "trading_reward"

    try:
        tree = ast.parse(source_code)
    except Exception:
        return "forecasting_pair"

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        init_args: list[str] = []
        forward_args: list[str] = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                init_args = [a.arg for a in item.args.args if a.arg != "self"]
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                forward_args = [a.arg for a in item.args.args if a.arg != "self"]
        if len(forward_args) >= 2:
            if any(arg.startswith("x_") for arg in forward_args):
                return "generative_pair"
            return "forecasting_pair"
        if "x_real" in init_args:
            return "generative_init_real_forward_fake"
        if len(forward_args) == 1:
            return "generative_single_fake"
    return "forecasting_pair"


class AssemblyGroundTruthError(Exception):
    """Raised when assembly cannot resolve a ground-truth contract."""


def _build_artifact_plan(
    dataset: "DatasetArtifact | None",
    interaction_model: str,
) -> dict[str, Any]:
    """Group manifest artifacts by role for evaluator assembly.

    Expected output names remain framework-controlled. ``has_ground_truth`` is
    diagnostic; consumers should use role lists or loader provenance.
    """
    payload_files: list[str] = []
    artifacts: list[dict[str, Any]] = []
    manifest_status = ""
    if dataset is not None:
        payload_files = list(dataset.payload_files)
        artifacts = [item for item in dataset.artifacts if isinstance(item, dict)]
        manifest_status = str(dataset.manifest_status or "")

    def _filter_by_role(role: str) -> list[dict[str, Any]]:
        return [
            dict(a)
            for a in artifacts
            if str(a.get("role", "")).strip() == role and str(a.get("path", "")).strip()
        ]

    # Role lists remain useful for validation and diagnostics.
    ground_truth_artifacts = _filter_by_role("ground_truth")
    reference_artifacts = _filter_by_role("reference")
    feature_artifacts = _filter_by_role("features")

    # Agents write to framework-defined filenames.
    if interaction_model == "forecasting":
        expected_agent_outputs = ["predictions.npy", "predictions.csv"]
    elif interaction_model == "generative":
        expected_agent_outputs = [
            "samples.npy",
            "samples.csv",
            "predictions.npy",
            "predictions.csv",
        ]
    else:
        expected_agent_outputs = []

    return {
        "dataset_payload_files": payload_files,
        "expected_agent_outputs": expected_agent_outputs,
        "ground_truth_artifacts": ground_truth_artifacts,
        "reference_artifacts": reference_artifacts,
        "feature_artifacts": feature_artifacts,
        "has_ground_truth": bool(ground_truth_artifacts or reference_artifacts),
        "manifest_status": manifest_status,
        "interaction_model": interaction_model,
    }


def _validate_artifact_plan_for_assembly(
    plan: dict[str, Any],
    *,
    task_id: str,
    dataset_name: str,
    dataset_data_dir: Path | None,
) -> None:
    """Reject unresolved manifests and missing declared target artifacts.

    A loader may derive ground truth from features when no target file exists.
    """
    status = str(plan.get("manifest_status", "")).strip()
    interaction_model = str(plan.get("interaction_model", "")).strip()
    gt_artifacts = plan.get("ground_truth_artifacts", []) or []
    ref_artifacts = plan.get("reference_artifacts", []) or []

    if status == "unresolved":
        raise AssemblyGroundTruthError(
            f"task={task_id} dataset={dataset_name!r} has manifest_status='unresolved' "
            "— the labeler could not confidently assign file roles. Inspect the "
            "dataset's manifest.json, fix the 'artifacts' list and flip "
            "manifest_status to 'labeled', then re-run assembly."
        )
    if status == "unlabeled_curated":
        raise AssemblyGroundTruthError(
            f"task={task_id} dataset={dataset_name!r} is a curated dataset "
            "without a labeled manifest. Author a manifest.json at the "
            "dataset's directory describing its file roles/formats, then "
            "re-run assembly. (Automated labeling of curated datasets is "
            "out of scope for this pipeline.)"
        )
    if dataset_data_dir is not None and dataset_data_dir.exists():
        for entry in list(gt_artifacts) + list(ref_artifacts):
            rel = str(entry.get("path", "")).strip()
            if not rel:
                continue
            target = dataset_data_dir / rel
            if not target.exists():
                raise AssemblyGroundTruthError(
                    f"task={task_id} dataset={dataset_name!r} manifest lists "
                    f"artifact path {rel!r} with role={entry.get('role')!r} but "
                    f"the file does not exist at {target}."
                )
    if (
        interaction_model == "forecasting"
        and not gt_artifacts
        and not ref_artifacts
    ):
        log_detail(
            logger,
            "%s task=%s dataset=%s forecasting task has no role=ground_truth "
            "artifact on disk; the loader will derive ground_truth from "
            "feature columns and record the choice in ground_truth_provenance",
            PHASE_TAG,
            task_id,
            dataset_name,
        )


def assemble_evaluator(
    task_id: str,
    task_title: str,
    rewards: list[dict[str, Any]],
    eval_root: Path,
    staging_dir: Path,
    interaction_model: str = "forecasting",
    dataset_payload_files: list[str] | None = None,
    expected_agent_outputs: list[str] | None = None,
    artifact_plan: dict[str, Any] | None = None,
    parents_depth: int = 5,
    generated_dir: Path | None = None,
    block_reward_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble evaluator.py from resolved generated and curated rewards.

    ``artifact_plan`` overrides legacy file/output arguments. Slugified
    ``block_reward_keys`` are omitted and reported as pruned.
    """
    import jinja2

    generated_root = generated_dir if generated_dir is not None else (eval_root / "generated")
    reward_bank_json_path = eval_root / "reward_bank.json"
    curated_imports: list[dict[str, str]] = []
    generated_imports: list[dict[str, str]] = []
    reward_specs: list[dict[str, Any]] = []
    unresolved_rewards: list[str] = []
    rewards_resolved: list[str] = []
    rewards_pruned: list[str] = []
    blocked_keys: set[str] = {
        str(k).strip()
        for k in (block_reward_keys or [])
        if str(k).strip()
    }
    reward_assembly_errors: dict[str, str] = {}
    produced_ctx_keys: set[str] = set()
    # Collapse paper aliases that resolve to the same metric class.
    seen_classes: set[str] = set()
    aliases_collapsed: list[tuple[str, str]] = []  # (paper_name, canonical_class)

    def _record_resolved(
        *,
        class_name: str,
        module_name: str,
        reward_name: str,
        class_source: str,
        resolution: dict[str, Any],
        is_generated: bool,
        entry: dict[str, Any],
    ) -> bool:
        """Validate and append a reward, recording assembly errors."""
        base_class = str(resolution.get("base_class", "")).strip() or "Loss"
        default_params: dict[str, Any] = dict(resolution.get("default_params", {}))
        entry_params = entry.get("params")
        if isinstance(entry_params, dict):
            final_params: dict[str, Any] = {**default_params, **entry_params}
        else:
            final_params = dict(default_params)

        # Revalidate Loss contracts before runtime binding.
        from openfinai_pipeline.corpus.contract_validator import (
            validate_reward_contract,
        )
        contract_violations = (
            validate_reward_contract(class_source)
            if class_source and base_class != "TradingReward"
            else []
        )
        # Reject unknown constructor overrides before scoring.
        init_names, forward_names_for_spec = _ast_class_method_params(class_source, class_name)
        _LOSS_BASE_KWARGS = {"name"}
        effective_init_names = init_names if init_names else _LOSS_BASE_KWARGS
        errors: list[str] = list(contract_violations)
        bad_param_kw = [
            kw for kw in final_params.keys() if kw not in effective_init_names
        ]
        if bad_param_kw:
            errors.append(
                f"params reference kwargs not in {class_name}.__init__: "
                f"{bad_param_kw} (has: {sorted(effective_init_names)})"
            )
        if errors:
            reward_assembly_errors[reward_name] = "; ".join(errors)
            log_detail(
                logger,
                "evaluator_assembly task=%s reward=%s validation_failed: %s",
                task_id, reward_name, errors[0],
            )
            return False

        # Runtime score keys use slugified class names. Track blocked valid
        # rewards as pruned, distinct from rewards that could not be resolved.
        if blocked_keys and _slugify(class_name) in blocked_keys:
            pruned_label = f"{reward_name} -> {class_name}"
            rewards_pruned.append(pruned_label)
            log_detail(
                logger,
                "evaluator_assembly task=%s reward=%s class=%s pruned_for_prediction_blind",
                task_id, reward_name, class_name,
            )
            return False

        # Key specs by canonical class name so aliases cannot schedule the same
        # metric twice; ``rewards_resolved`` retains aliases for auditing.
        if class_name in seen_classes:
            aliases_collapsed.append((reward_name, class_name))
            log_detail(
                logger,
                "evaluator_assembly task=%s reward=%s aliased to existing "
                "class=%s; not adding duplicate spec/import",
                task_id, reward_name, class_name,
            )
            return True
        seen_classes.add(class_name)

        if is_generated:
            generated_imports.append(
                {
                    "module_name": module_name,
                    "class_name": class_name,
                }
            )
        else:
            curated_imports.append(
                {
                    "module_name": module_name,
                    "class_name": class_name,
                }
            )
        # Reuse parsed signature names in renderers and context validation.
        from openfinai_pipeline.prompts.canonical_ctx import ALL_CANONICAL_NAMES
        init_canonical = sorted(set(init_names) & ALL_CANONICAL_NAMES)
        forward_canonical = sorted(set(forward_names_for_spec) & ALL_CANONICAL_NAMES)
        reward_specs.append(
            {
                "reward_key": _slugify(class_name),
                "class_name": class_name,
                "base_class": base_class,
                "final_params": final_params,
                "init_canonical_names": init_canonical,
                "forward_canonical_names": forward_canonical,
            }
        )
        for name in init_canonical:
            produced_ctx_keys.add(name)
        for name in forward_canonical:
            produced_ctx_keys.add(name)
        return True

    for reward_entry in rewards:
        reward_name = str(reward_entry.get("name", "")).strip()
        if not reward_name:
            continue

        resolved = False
        resolution = _resolve_reward_reference(
            reward_name,
            reward_description=str(reward_entry.get("description", "")).strip(),
            reward_script_path=str(reward_entry.get("reward_script_path", "")).strip(),
            generated_root=generated_root,
            reward_bank_json_path=reward_bank_json_path,
            resolved_class_name=str(
                reward_entry.get("resolved_class_name", "")
            ).strip(),
        )
        if resolution is not None:
            source = str(resolution.get("source", "")).strip()
            module_name = str(resolution.get("module_name", "")).strip()
            class_name = str(resolution.get("class_name", "")).strip()
            class_source = str(resolution.get("code", ""))
            if module_name and class_name:
                if _record_resolved(
                    class_name=class_name,
                    module_name=module_name,
                    reward_name=reward_name,
                    class_source=class_source,
                    resolution=resolution,
                    is_generated=source.startswith("generated_"),
                    entry=reward_entry,
                ):
                    tag = "generated" if source.startswith("generated_") else "curated"
                    rewards_resolved.append(f"{reward_name} ({tag})")
                    resolved = True

        if not resolved:
            unresolved_rewards.append(reward_name)
            log_detail(
                logger,
                "evaluator_assembly task=%s reward=%s UNRESOLVED",
                task_id,
                reward_name,
            )

    class_name_prefix = task_id.replace("_", " ").title().replace(" ", "")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )
    template = env.get_template("evaluator_assembled.py.j2")

    # A manifest plan overrides legacy file and output arguments.
    if artifact_plan is not None:
        plan_payload_files = list(artifact_plan.get("dataset_payload_files", []) or [])
        plan_expected_outputs = list(artifact_plan.get("expected_agent_outputs", []) or [])
        plan_manifest_status = str(artifact_plan.get("manifest_status", "") or "")
    else:
        plan_payload_files = list(dataset_payload_files or [])
        plan_expected_outputs = list(expected_agent_outputs or [])
        plan_manifest_status = ""

    # Bundle reward dependencies beside the evaluator.
    staging_dir.mkdir(parents=True, exist_ok=True)
    bundled_imports: list[dict[str, str]] = []
    if curated_imports or generated_imports:
        from openfinai_pipeline.benchmark.install.reward_bundler import (
            RewardBundlerError,
            bundle_used_rewards,
        )
        try:
            bundle_result = bundle_used_rewards(
                curated_imports=curated_imports,
                generated_imports=generated_imports,
                curated_root=Path(eval_root),
                generated_root=Path(generated_root) if generated_root.exists() else None,
                out_path=staging_dir / "task_rewards.py",
            )
            log_detail(
                logger,
                "task_rewards bundled task=%s out=%s classes=%d helpers=%d",
                task_id,
                bundle_result.out_path,
                len(bundle_result.classes_bundled),
                len(bundle_result.helpers_bundled),
            )
        except RewardBundlerError as exc:
            # Let the caller decide whether assembly errors reject the task.
            reward_assembly_errors["__bundle__"] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "evaluator_assembly task=%s bundle_failed: %s", task_id, exc
            )
            return {
                "success": False,
                "rewards_resolved": len(rewards_resolved),
                "resolved_reward_names": [spec["class_name"] for spec in reward_specs],
                "resolved_reward_keys": [spec["reward_key"] for spec in reward_specs],
                "rewards_unresolved": unresolved_rewards,
                "rewards_pruned": list(rewards_pruned),
                "reward_assembly_errors": reward_assembly_errors,
                "produced_ctx_keys": sorted(produced_ctx_keys),
                "aliases_collapsed": list(aliases_collapsed),
                "path": "",
            }
        for entry in (*curated_imports, *generated_imports):
            bundled_imports.append(
                {
                    "class_name": entry["class_name"],
                }
            )

    from openfinai_pipeline.prompts.canonical_ctx import ALL_CANONICAL_NAMES
    rendered = template.render(
        title=task_title,
        class_name=class_name_prefix,
        interaction_model=interaction_model,
        dataset_payload_files=plan_payload_files,
        expected_agent_outputs=plan_expected_outputs,
        manifest_status=plan_manifest_status,
        bundled_imports=bundled_imports,
        reward_specs=reward_specs,
        canonical_ctx_keys=sorted(ALL_CANONICAL_NAMES),
        unresolved_rewards=unresolved_rewards,
        rewards_resolved=rewards_resolved,
    )

    out_path = staging_dir / "evaluator.py"
    out_path.write_text(rendered, encoding="utf-8")
    log_detail(
        logger,
        "evaluator_assembled task=%s path=%s resolved=%d unresolved=%d errors=%d aliases_collapsed=%d pruned=%d",
        task_id,
        out_path,
        len(rewards_resolved),
        len(unresolved_rewards),
        len(reward_assembly_errors),
        len(aliases_collapsed),
        len(rewards_pruned),
    )
    if aliases_collapsed:
        for paper_name, canonical in aliases_collapsed:
            log_detail(
                logger,
                "evaluator_assembly task=%s alias_collapsed paper_name=%s -> class=%s",
                task_id, paper_name, canonical,
            )

    return {
        "success": len(unresolved_rewards) == 0 or len(rewards_resolved) > 0,
        "rewards_resolved": len(rewards_resolved),
        "resolved_reward_names": [spec["class_name"] for spec in reward_specs],
        "resolved_reward_keys": [spec["reward_key"] for spec in reward_specs],
        "rewards_unresolved": unresolved_rewards,
        "rewards_pruned": list(rewards_pruned),
        "reward_assembly_errors": reward_assembly_errors,
        "produced_ctx_keys": sorted(produced_ctx_keys),
        "aliases_collapsed": list(aliases_collapsed),
        "path": str(out_path),
    }


# Task candidate loading


def _dataset_short_suffix(dataset_name: str) -> str:
    """Extract the asset token from a conventional dataset name, or slugify it."""
    parts = (dataset_name or "").split("_")
    if len(parts) >= 3:
        candidate = slugify(parts[1])
        if candidate and candidate != "dataset":
            return candidate[:30]
    return slugify(dataset_name)[:30]


def _fan_out_per_dataset(base: TaskCandidate) -> list[TaskCandidate]:
    """Emit one uniquely named candidate per dataset.

    TODO(cross-sectional): retain joint datasets once corpus summaries expose
    whether a group is parallel or joint.
    """
    datasets = base.datasets or []
    if len(datasets) <= 1:
        return [base]

    # Prefer the asset token in {Provider}_{Asset}_{...} names.
    suffixes = [_dataset_short_suffix(str(ds.get("name", ""))) for ds in datasets]

    # On ambiguity, use full slugs consistently across siblings.
    if len(set(suffixes)) != len(suffixes) or any(not s for s in suffixes):
        suffixes = [slugify(str(ds.get("name", "")))[:30] for ds in datasets]

    # Add stable discriminators if truncated or duplicate names still collide.
    seen: set[str] = set()
    unique_suffixes: list[str] = []
    for i, suffix in enumerate(suffixes):
        final = suffix
        attempt = 0
        while final in seen:
            attempt += 1
            final = (
                f"{suffix}_ds{i}"
                if attempt == 1
                else f"{suffix}_ds{i}_{attempt}"
            )
        seen.add(final)
        unique_suffixes.append(final)

    siblings: list[TaskCandidate] = []
    for ds, suffix in zip(datasets, unique_suffixes):
        ds_name = str(ds.get("name", "")).strip()
        siblings.append(
            replace(
                base,
                task_id=f"{base.task_id}__{suffix}",
                title=f"{base.title} ({ds_name})" if ds_name else base.title,
                datasets=[ds],
            )
        )
    return siblings or [base]


def load_task_candidates(research_root: str | Path) -> list[TaskCandidate]:
    """Load scoped papers in global order and fan out multi-dataset tasks."""
    root = Path(research_root)
    candidates: list[TaskCandidate] = []

    scope_paper_pairs: list[tuple[Path, Path]] = []
    for scope_dir in [p for p in root.iterdir() if p.is_dir()]:
        for paper_dir in [
            p
            for p in scope_dir.iterdir()
            if p.is_dir() and p.name.startswith("paper")
        ]:
            scope_paper_pairs.append((scope_dir, paper_dir))
    scope_paper_pairs.sort(
        key=lambda pair: paper_global_sort_key(f"{pair[0].name}/{pair[1].name}")
    )

    for scope_dir, paper_dir in scope_paper_pairs:
        paper_path = paper_dir / "paper.json"
        if not paper_path.exists():
            continue

        try:
            paper_data = json.loads(paper_path.read_text(encoding="utf-8"))
            problem = extract_summary_payload(paper_data)
            if not isinstance(problem, dict):
                continue
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "%s skip paper_dir=%s bad_paper_json=%s",
                PHASE_TAG,
                paper_dir.name,
                exc,
            )
            continue

        # Curated-routed papers are already bound to task overlays.
        task_record = paper_data.get("task")
        if (
            isinstance(task_record, dict)
            and str(task_record.get("kind", "")).strip().lower() == "curated_routed"
        ):
            continue

        paper_abstract = ""
        scope_id = str(paper_data.get("scope_id", "")).strip() or scope_dir.name
        paper_id = ""
        paper_obj = paper_data.get("paper", {})
        if isinstance(paper_obj, dict):
            paper_abstract = paper_obj.get("abstract", "")
            paper_id = paper_obj.get("paper_id", "")

        pdf_path = paper_dir / "source.pdf"
        raw_task_name = problem.get("task_name", "").strip()
        task_slug = (
            slugify(raw_task_name)
            if raw_task_name
            else slugify(problem.get("ml_task_summary", paper_dir.name))
        )
        task_slug = task_slug[:60] or paper_dir.name
        base_candidate = TaskCandidate(
            task_dir=str(paper_dir),
            task_id=task_slug,
            title=raw_task_name
            or problem.get("ml_task_summary", paper_dir.name),
            scope_id=scope_id,
            paper_id=paper_id,
            ml_task_summary=problem.get("ml_task_summary", ""),
            experiments=problem.get("experiments", ""),
            datasets=problem.get("datasets", []),
            rewards=problem.get("metrics", []),
            task_family=_normalize_task_family(problem.get("task_family", "")),
            paper_abstract=paper_abstract,
            pdf_path=str(pdf_path) if pdf_path.exists() else None,
        )
        candidates.extend(_fan_out_per_dataset(base_candidate))

    try:
        from_display = Path(root).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        from_display = Path(root).as_posix()
    logger.info(
        "%s   load task candidates count=%d from=%s",
        PHASE_TAG,
        len(candidates),
        from_display,
    )
    return candidates


# Deduplication


def can_reuse_existing_task(
    candidate: TaskCandidate,
    existing_tasks: list[dict[str, str]],
    llm: LLMService,
) -> tuple[bool, str, str]:
    """
    LLM-based task dedup.  Returns (is_duplicate, matched_task_id, reasoning).
    """
    if not existing_tasks:
        return False, "", "no_existing_tasks"

    prompt = build_task_dedup_prompt(
        candidate_name=candidate.title,
        candidate_description=candidate.ml_task_summary,
        existing_tasks=existing_tasks,
    )
    payload = llm.complete_or_fallback(
        prompt,
        task_dedup_schema(),
        fallback={
            "is_duplicate": False,
            "matched_task_id": "",
            "reasoning": "fallback_no_dedup",
        },
    )

    is_dup = bool(payload.get("is_duplicate", False))
    matched = str(payload.get("matched_task_id", "")).strip()
    reason = str(payload.get("reasoning", "")).strip()

    if not is_dup or not matched:
        return False, "", reason or "not_duplicate"

    valid_ids = {t.get("task_id", "") for t in existing_tasks}
    if matched not in valid_ids:
        return False, "", "llm_matched_id_not_found"

    return True, matched, reason or "duplicate"


# Code generation


def _build_enriched_task_info(
    candidate: TaskCandidate,
    phase_artifacts: PhaseArtifacts | None = None,
    *,
    resolved_reward_names: list[str] | None = None,
) -> str:
    """Build token-capped task context from resolved Phase 2/3 artifacts.

    Payload paths describe schemas only; resolved rewards are authoritative.
    """
    parts = [
        f"Task ID: {candidate.task_id}",
        f"Title: {candidate.title}",
        f"Task Family: {candidate.task_family or 'unknown'}",
        f"Summary: {candidate.ml_task_summary}",
        f"Experiments: {candidate.experiments}",
        f"Paper ID: {candidate.paper_id}",
    ]

    if phase_artifacts is None:
        # Fall back to raw paper-summary data.
        parts.append(f"Datasets: {json.dumps(candidate.datasets)}")
        parts.append(f"Rewards: {json.dumps(candidate.rewards)}")
        return "\n".join(parts)

    total_code_chars = sum(len(d.download_script) for d in phase_artifacts.datasets)
    total_code_chars += sum(len(m.code) for m in phase_artifacts.rewards)
    compact_mode = total_code_chars > 8000

    parts.append("")
    parts.append("=== Datasets ===")
    parts.append(
        "Note: at install time the slicer materialises a single "
        "`/data/dataset.h5` plus a verifier-only "
        "`/eval-data/test_ground_truth.h5`. The original payload files "
        "below are NOT shipped to the agent — they are listed here ONLY "
        "as a guide to the column / schema semantics you should describe "
        "in `schema_description` for instruction.md."
    )
    for i, ds in enumerate(phase_artifacts.datasets):
        parts.append(f"Dataset: {ds.name}")
        if ds.downloaded_dataset_description:
            parts.append(
                "  Downloaded dataset description: "
                f"{ds.downloaded_dataset_description}"
            )
        parts.append(f"  Approved: {ds.approved}")
        if ds.payload_files:
            parts.append(
                "  Source payload files (schema reference only, NOT "
                f"present at /data/ at runtime): {json.dumps(ds.payload_files)}"
            )
        if ds.preview:
            parts.append(
                "  Payload preview (use this together with the downloaded "
                "dataset description to describe the feature schema in "
                "`schema_description`; do NOT reimplement the parser): "
                f"{json.dumps(ds.preview, ensure_ascii=False)}"
            )
        if ds.download_script:
            if compact_mode and i > 0:
                parts.append(
                    "  (download script omitted for brevity)"
                )
            else:
                parts.append(
                    "  Working Phase-2 download script (reference only; do not execute it at runtime):"
                )
                parts.append(f"    ```python\n{ds.download_script}\n    ```")

    # Include summary datasets not matched in Phase 2.
    matched_names = {d.name.lower() for d in phase_artifacts.datasets}
    for ds_entry in candidate.datasets:
        ds_name = ds_entry.get("name", "")
        if ds_name.lower() not in matched_names:
            parts.append(f"Dataset: {ds_name} (no Phase 2 download script available)")
            parts.append(f"  Description: {ds_entry.get('description', '')}")

    # Only assembled rewards belong in generated instructions.
    parts.append("")
    parts.append("=== Resolved metrics (these are what the evaluator will compute) ===")
    resolved_set = {n for n in (resolved_reward_names or []) if n}
    if resolved_set:
        for ma in phase_artifacts.rewards:
            if ma.class_name and ma.class_name not in resolved_set:
                continue
            parts.append(f"Reward: {ma.name}")
            parts.append(f"  Source: {ma.source}")
            if ma.module_name or ma.class_name:
                parts.append(
                    "  Reward bank identity: "
                    f"module={ma.module_name or '(unknown)'} "
                    f"class={ma.class_name or '(unknown)'}"
                )
            if ma.base_class:
                parts.append(f"  Base class: {ma.base_class}")
            if ma.paper_metric_description:
                parts.append(
                    f"  Paper metric description: {ma.paper_metric_description}"
                )
            if ma.code:
                parts.append("  Reference implementation:")
                parts.append(f"    ```python\n{ma.code}\n    ```")
        unresolved_paper_names = sorted(
            {ma.name for ma in phase_artifacts.rewards if ma.class_name not in resolved_set}
        ) + sorted({um.get("name", "") for um in phase_artifacts.unmatched_rewards if um.get("name")})
        unresolved_paper_names = [n for n in unresolved_paper_names if n]
        if unresolved_paper_names:
            parts.append("")
            parts.append(
                "=== Paper-claimed metrics that did NOT resolve "
                "(do NOT mention them in instruction.md prose) ==="
            )
            parts.extend(f"- {n}" for n in unresolved_paper_names)
    else:
        # Legacy callers without a resolved set receive every reward.
        for ma in phase_artifacts.rewards:
            parts.append(f"Reward: {ma.name}")
            parts.append(f"  Source: {ma.source}")
            if ma.module_name or ma.class_name:
                parts.append(
                    "  Reward bank identity: "
                    f"module={ma.module_name or '(unknown)'} "
                    f"class={ma.class_name or '(unknown)'}"
                )
            if ma.base_class:
                parts.append(f"  Base class: {ma.base_class}")
            if ma.paper_metric_description:
                parts.append(
                    f"  Paper metric description: {ma.paper_metric_description}"
                )
            if ma.code:
                parts.append("  Reference implementation:")
                parts.append(f"    ```python\n{ma.code}\n    ```")
        for um in phase_artifacts.unmatched_rewards:
            parts.append(
                f"Reward: {um['name']} (unmatched in Phase 3 — may or may not "
                "be in the final assembled evaluator; check the manifest)"
            )
            parts.append(f"  Description: {um.get('description', '')}")

    parts.append("")
    parts.append("=== Evaluator / Agent Output Contract ===")
    if candidate.task_family == "forecasting":
        parts.append(
            "The agent must write `predictions.npy` or `predictions.csv`. "
            "The packaged evaluator is called as "
            "`score(predictions, ground_truth, weights=None, reward_output=None, **kwargs)`. "
            "These canonical filenames + the evaluator contract are rendered "
            "into instruction.md by the pipeline — do NOT restate them in the "
            "prose slots."
        )
    elif candidate.task_family == "generative":
        parts.append(
            "The agent must write `samples.npy` or `samples.csv`. "
            "Legacy aliases `predictions.npy` and `predictions.csv` are also accepted. "
            "The packaged evaluator is called as "
            "`score(predictions, ground_truth, weights=None, reward_output=None, **kwargs)`. "
            "These canonical filenames + the evaluator contract are rendered "
            "into instruction.md by the pipeline — do NOT restate them in the "
            "prose slots."
        )
    else:
        parts.append(
            "The packaged evaluator is called as "
            "`score(predictions, ground_truth, weights=None, reward_output=None, **kwargs)`. "
            "Follow the task-specific output expectations described above."
        )

    return "\n".join(parts)


def generate_task_code(
    candidate: TaskCandidate,
    llm: LLMService,
    paper_context: str,
    base_task_source: str,
    example_task_source: str,
    previous_code: GeneratedCode | None = None,
    error_output: str = "",
    phase_artifacts: PhaseArtifacts | None = None,
    template_reference: str = "",
    interaction_model: str = "gym",
    resolved_reward_names: list[str] | None = None,
    task_rewards_path: Path | None = None,
    expected_outputs: list[str] | None = None,
    split_policy: str = "",
    has_held_out_test_gt: bool = True,
) -> GeneratedCode:
    """Generate task.py and render instruction.md from its prose slots.

    Forecasting selects ForecastingTask and generative selects GenerativeTask;
    other interaction models are rejected. Retries receive prior errors.
    """
    resolved_reward_names = list(resolved_reward_names or [])
    expected_outputs = list(expected_outputs or [])
    task_info = _build_enriched_task_info(
        candidate,
        phase_artifacts,
        resolved_reward_names=resolved_reward_names,
    )

    if previous_code and error_output:
        # Retries return task.py and only changed prose slots.
        prose_block = (
            f"--- context_prose ---\n{previous_code.context_prose}\n\n"
            f"--- objective_prose ---\n{previous_code.objective_prose}\n\n"
            f"--- schema_description ---\n{previous_code.schema_description}"
        )
        code_summary = (
            f"--- task.py ---\n{previous_code.task_py}\n\n"
            f"{prose_block}"
        )
        prompt = build_fix_errors_prompt(
            task_info,
            code_summary,
            error_output,
            base_task_source=base_task_source,
            paper_context=paper_context,
            template_reference=template_reference,
        )
        schema = task_fix_schema()
        payload = llm.complete_structured(prompt, schema)
        partial = _parse_generated_code(payload)
        merged = GeneratedCode(
            task_py=partial.task_py or previous_code.task_py,
            context_prose=partial.context_prose or previous_code.context_prose,
            objective_prose=partial.objective_prose or previous_code.objective_prose,
            schema_description=partial.schema_description or previous_code.schema_description,
        )
    else:
        prompt = build_implementation_prompt(
            task_info,
            base_task_source,
            example_task_source,
            paper_context,
            template_reference=template_reference,
            interaction_model=interaction_model,
        )
        schema = task_generation_schema()
        payload = llm.complete_structured(prompt, schema)
        merged = _parse_generated_code(payload)

    # Render now so retries can inspect the complete generated package.
    metric_specs = build_metric_specs_for_instruction(
        resolved_reward_names=resolved_reward_names,
        task_rewards_path=task_rewards_path,
    )
    if interaction_model in ("forecasting", "generative"):
        try:
            merged.instruction_md = render_instruction_md(
                candidate=candidate,
                interaction_model=interaction_model,
                expected_outputs=expected_outputs,
                metric_specs=metric_specs,
                context_prose=merged.context_prose,
                objective_prose=merged.objective_prose,
                schema_description=merged.schema_description,
                split_policy=split_policy,
                has_held_out_test_gt=has_held_out_test_gt,
            )
        except Exception as exc:  # noqa: BLE001 — surface as a soft failure
            # Let smoke validation report a missing rendered instruction.
            logger.warning(
                "render_instruction_md failed for task_id=%s: %s",
                candidate.task_id,
                exc,
            )
            merged.instruction_md = ""
    return merged


def _parse_generated_code(payload: dict[str, Any]) -> GeneratedCode:
    """Extract task.py and instruction prose from an LLM payload."""

    def _clean(val: Any) -> str:
        s = str(val or "")
        # Strip optional Markdown fences.
        if s.startswith("```"):
            lines = s.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines)
        # Reject placeholder-only responses.
        if s.strip() in ("...", "pass"):
            return ""
        return s

    return GeneratedCode(
        task_py=_clean(payload.get("task_py", "")),
        context_prose=_clean(payload.get("context_prose", "")),
        objective_prose=_clean(payload.get("objective_prose", "")),
        schema_description=_clean(payload.get("schema_description", "")),
        evaluator_py=_clean(payload.get("evaluator_py", "")),
    )


# Static analysis


def static_check_code(code: GeneratedCode, sandbox: Sandbox) -> list[str]:
    """Run syntax + dangerous import + import resolution checks on all files."""
    violations: list[str] = []
    if not code.task_py.strip():
        violations.append("task.py: file is empty — generate the full implementation")
    files_to_check = [
        ("task.py", code.task_py),
    ]
    # Only LLM-authored files receive generation-specific checks.
    if code.evaluator_py:
        files_to_check.append(("evaluator.py", code.evaluator_py))
    for label, src in files_to_check:
        ok, err = sandbox.syntax_check(src)
        if not ok:
            violations.append(f"{label}: {err}")
        imps = sandbox.scan_dangerous_imports(src)
        for v in imps:
            violations.append(f"{label}: {v}")
        import_errs = sandbox.validate_openfingym_imports(src)
        for v in import_errs:
            violations.append(f"{label}: {v}")
    return violations


# Staging & installation


def write_staging(staging_dir: Path, code: GeneratedCode) -> None:
    """Stage generated files without replacing an assembled evaluator.

    Deterministic tests and the manifest are staged separately.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "task.py").write_text(code.task_py, encoding="utf-8")
    if code.instruction_md.strip():
        (staging_dir / "instruction.md").write_text(
            code.instruction_md,
            encoding="utf-8",
        )
    evaluator_path = staging_dir / "evaluator.py"
    if not evaluator_path.exists() and code.evaluator_py:
        evaluator_path.write_text(code.evaluator_py, encoding="utf-8")
    (staging_dir / "__init__.py").write_text(code.init_py, encoding="utf-8")


def render_instruction_md(
    *,
    candidate: TaskCandidate,
    interaction_model: str,
    expected_outputs: list[str],
    metric_specs: list[dict[str, str]],
    context_prose: str,
    objective_prose: str,
    schema_description: str,
    split_policy: str = "",
    has_held_out_test_gt: bool = True,
) -> str:
    """Render instruction.md from fixed contracts and three prose slots.

    ``has_held_out_test_gt`` distinguishes verifier-only targets from shared
    references. Strict template variables make missing contract data fail fast.
    """
    title = _humanise_task_title(candidate)
    raw = (candidate.task_id or "").replace("-", "_")
    class_name = "".join(word.capitalize() for word in raw.split("_")) or "GeneratedTask"
    expected_list = list(expected_outputs or [])
    if interaction_model == "forecasting" and not expected_list:
        expected_list = ["predictions.npy", "predictions.csv"]
    elif interaction_model == "generative" and not expected_list:
        expected_list = ["samples.npy", "samples.csv"]
    default_output = expected_list[0] if expected_list else "predictions.npy"

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    template = env.get_template("instruction.md.j2")
    return template.render(
        title=title,
        class_name=class_name,
        interaction_model=interaction_model,
        expected_outputs=expected_list,
        default_output=default_output,
        metric_specs=list(metric_specs),
        context_prose=(context_prose or "").strip()
        or "(No paper-context prose was provided; see manifest + paper for details.)",
        objective_prose=(objective_prose or "").strip()
        or "(No objective prose was provided; see manifest for the resolved metric set.)",
        schema_description=(schema_description or "").strip()
        or "(No schema prose was provided. Inspect `dataset.h5` directly for the bundled feature shapes.)",
        split_policy=(split_policy or "").strip(),
        has_held_out_test_gt=bool(has_held_out_test_gt),
    )


def _humanise_task_title(candidate: TaskCandidate) -> str:
    """Choose a title and humanize identifier-like underscores."""
    source = (candidate.title or "").strip() or (candidate.task_id or "").strip()
    if not source:
        return "Generated Task"
    if "_" in source and not any(ch.isspace() for ch in source):
        return source.replace("_", " ").strip()
    return source


def _render_harbor_task_toml(candidate: TaskCandidate, interaction_model: str) -> str:
    keywords = [candidate.scope_id, interaction_model]
    keywords_literal = ", ".join(f'"{kw}"' for kw in keywords if kw)
    description = (candidate.ml_task_summary or candidate.title).replace("\n", " ").replace('"', '\\"').strip()
    return (
        'version = "1.0"\n\n'
        "[metadata]\n"
        f'name = "{candidate.scope_id}/{candidate.task_id}"\n'
        f'description = "{description}"\n'
        'difficulty = "medium"\n'
        'domain = "financial"\n'
        f"keywords = [{keywords_literal}]\n\n"
        "[agent]\n"
        "timeout_sec = 600\n\n"
        "[verifier]\n"
        "timeout_sec = 600\n\n"
        "[environment]\n"
        "cpus = 2\n"
        "memory_mb = 4096\n"
        # Networking carries submissions; verifier mounts isolate held-out targets.
        "allow_internet = true\n"
        # Task data is mounted into this shared registry image at launch.
        'docker_image = "nihao0630/openfinai-base:v1"\n'
    )


def _render_harbor_test_sh(candidate: TaskCandidate) -> str:
    return (
        "#!/bin/bash\n"
        "set -e\n"
        "mkdir -p /logs/verifier\n\n"
        f"python /data/{_EXPLICIT_RUNNER_NAME} \\\n"
        "    --evaluator-path /data/evaluator.py \\\n"
        "    --data-dir /data \\\n"
        "    --eval-data-dir /eval-data \\\n"
        "    --reward-output /logs/verifier/reward.json\n"
    )


def _render_harbor_dockerfile() -> str:
    # Harbor mounts task data into this shared image.
    return (
        "FROM nihao0630/openfinai-base:v1\n\n"
        "WORKDIR /workspace\n"
    )


_STATIC_TESTS_DIR = Path(__file__).resolve().parents[1] / "static_tests"

_STATIC_TEST_FILENAME = "test_task.py"
_STATIC_TEST_HELPERS_FILENAME = "_test_helpers.py"
_TASK_MANIFEST_FILENAME = "manifest.json"


def _canonical_class_name(task_id: str) -> str:
    """Return the task/evaluator contract's PascalCase class prefix."""
    return task_id.replace("_", " ").title().replace(" ", "")


def _stage_static_test_files(staging_dir: Path, interaction_model: str) -> None:
    """Stage the forecasting or generative contract-test pair."""
    if interaction_model not in ("forecasting", "generative"):
        raise ValueError(
            "_stage_static_test_files only supports interaction_model in "
            "{'forecasting','generative'}; the auto-pipeline does not stage "
            f"deterministic tests for {interaction_model!r}"
        )
    src_test = _STATIC_TESTS_DIR / f"test_task_{interaction_model}.py"
    src_helpers = _STATIC_TESTS_DIR / _STATIC_TEST_HELPERS_FILENAME
    if not src_test.exists():
        raise FileNotFoundError(
            f"static test file missing: {src_test}; the static_tests/ "
            "package is part of the codebase and must ship with the pipeline."
        )
    if not src_helpers.exists():
        raise FileNotFoundError(
            f"static test helpers missing: {src_helpers}"
        )
    shutil.copy2(src_test, staging_dir / _STATIC_TEST_FILENAME)
    shutil.copy2(src_helpers, staging_dir / _STATIC_TEST_HELPERS_FILENAME)


def _write_task_manifest(
    staging_dir: Path,
    candidate: TaskCandidate,
    interaction_model: str,
    resolved_reward_names: list[str],
    *,
    has_held_out_test_gt: bool = False,
    split_policy: str = "",
    ground_truth_provenance: dict[str, Any] | None = None,
) -> Path:
    """Write the task contract consumed by tests and the verifier.

    Held-out and split fields are staging placeholders later replaced with
    slicer results. Ground-truth provenance remains an installed audit record.
    """
    class_name = _canonical_class_name(candidate.task_id)
    expected_score_keys = sorted(
        {_slugify(name) for name in resolved_reward_names if name} | {"reward"}
    )
    if interaction_model == "forecasting":
        expected_outputs = ["predictions.npy", "predictions.csv"]
    elif interaction_model == "generative":
        expected_outputs = [
            "samples.npy",
            "samples.csv",
            "predictions.npy",
            "predictions.csv",
        ]
    else:
        raise ValueError(
            "_write_task_manifest only supports interaction_model in "
            f"{{'forecasting','generative'}}; got {interaction_model!r}"
        )
    payload = {
        "schema_version": 2,
        "task_id": candidate.task_id,
        "scope_id": candidate.scope_id,
        "interaction_model": interaction_model,
        "class_name": class_name,
        "evaluator_class_name": f"{class_name}Evaluator",
        "version": "0.1.0",
        "resolved_reward_names": list(resolved_reward_names),
        "expected_score_keys": expected_score_keys,
        "expected_agent_output_filenames": expected_outputs,
        "has_held_out_test_gt": bool(has_held_out_test_gt),
        "split_policy": split_policy or "",
        "ground_truth_provenance": (
            dict(ground_truth_provenance) if ground_truth_provenance else None
        ),
    }
    out = staging_dir / _TASK_MANIFEST_FILENAME
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def write_deterministic_harbor_assets(
    staging_dir: Path,
    candidate: TaskCandidate,
    phase_artifacts: PhaseArtifacts | None,
    resolved_reward_names: list[str],
    interaction_model: str,
    *,
    ground_truth_provenance: dict[str, Any] | None = None,
) -> None:
    if interaction_model not in ("forecasting", "generative"):
        raise ValueError(
            "write_deterministic_harbor_assets only supports interaction_model "
            "in {'forecasting','generative'}; the auto-pipeline does not "
            f"emit trading/gym/realtime tasks. Got: {interaction_model!r}"
        )
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "task.toml").write_text(
        _render_harbor_task_toml(candidate, interaction_model),
        encoding="utf-8",
    )
    tests_dir = staging_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_script = tests_dir / "test.sh"
    test_script.write_text(_render_harbor_test_sh(candidate), encoding="utf-8")
    test_script.chmod(0o755)
    # Contract tests stay in staging; the manifest is installed.
    _stage_static_test_files(staging_dir, interaction_model)
    _write_task_manifest(
        staging_dir,
        candidate,
        interaction_model,
        resolved_reward_names,
        ground_truth_provenance=ground_truth_provenance,
    )


def _copy_dataset_payloads_to_environment(
    dataset: DatasetArtifact | None,
    *,
    target_data_dir: Path,
    datasets_root: Path,
    curated_datasets_root: Path,
) -> None:
    if dataset is None or not dataset.payload_files:
        return

    if dataset.source_paths:
        source_pairs = [
            (Path(src_path), Path(rel_path))
            for src_path, rel_path in zip(dataset.source_paths, dataset.payload_files)
        ]
    else:
        source_root = (
            curated_datasets_root if dataset.source_kind == "curated" else datasets_root
        )
        source_pairs = []
        for rel_path in dataset.payload_files:
            rel = Path(str(rel_path))
            src = source_root / rel
            if not src.exists() and dataset.asset_dir_name and dataset.data_dir_name:
                src = datasets_root / dataset.asset_dir_name / dataset.data_dir_name / rel
            source_pairs.append((src, rel))

    for src, rel in source_pairs:
        if not src.exists():
            continue
        dst = target_data_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _installed_task_source_candidates(task_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    top_level = task_dir / "task.py"
    flat_path = task_dir / "environment" / "data" / "task.py"
    if top_level.exists():
        candidates.append(top_level)
    if flat_path.exists():
        candidates.append(flat_path)
    return candidates


def _explicit_runner_source_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "openfinai_harbor"
        / "eval"
        / _EXPLICIT_RUNNER_NAME
    )


def _harbor_submit_source_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "openfinai_harbor"
        / "verifier"
        / "submit.py"
    )


def install_task(
    scope_id: str,
    task_id: str,
    staging_dir: Path,
    tasks_root: Path,
    *,
    phase_artifacts: PhaseArtifacts | None,
    datasets_root: Path,
    curated_datasets_root: Path,
    loader_ready: bool = False,
    pdf_path: str | Path | None = None,
    installed_task_ids: set[str] | None = None,
) -> Path:
    """Install a Harbor task under ``<tasks_root>/<scope_id>/<task_id>``.

    Agent-readable files live in environment/data and held-out targets in
    environment/eval-data; Harbor mounts them separately. Static contract tests
    remain in staging. An available source PDF is copied for reviewers.
    """
    dest = tasks_root / scope_id / task_id
    # Prevent two candidates in one run from sharing a task_id. ``overwrite``
    # may still replace tasks from earlier runs.
    if installed_task_ids is not None and task_id in installed_task_ids:
        raise RuntimeError(
            f"install_task refuses to overwrite task_id={task_id!r} at "
            f"{dest} — another candidate already installed under the same "
            "id during this run. This indicates a task_id collision that "
            "the fan-out / candidate-load layer failed to prevent."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)

    (dest / "tests").mkdir(parents=True, exist_ok=True)
    data_dir = dest / "environment" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    eval_data_dir = dest / "environment" / "eval-data"

    for name in ("instruction.md", "task.toml"):
        src = staging_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)

    if pdf_path:
        pdf_src = Path(pdf_path)
        if pdf_src.exists() and pdf_src.is_file():
            try:
                shutil.copy2(pdf_src, dest / f"{task_id}.pdf")
            except OSError as exc:
                logger.warning(
                    "%s pdf_copy_failed task_id=%s src=%s error=%s",
                    PHASE_TAG,
                    task_id,
                    pdf_src,
                    exc,
                )
        else:
            logger.warning(
                "%s pdf_copy_skipped task_id=%s reason=missing src=%s",
                PHASE_TAG,
                task_id,
                pdf_src,
            )

    test_script = staging_dir / "tests" / "test.sh"
    if test_script.exists():
        shutil.copy2(test_script, dest / "tests" / "test.sh")
        (dest / "tests" / "test.sh").chmod(0o755)

    evaluator_src = staging_dir / "evaluator.py"
    if evaluator_src.exists():
        shutil.copy2(evaluator_src, data_dir / "evaluator.py")
    # Install the assembled evaluator's optional reward bundle.
    task_rewards_src = staging_dir / "task_rewards.py"
    if task_rewards_src.exists():
        shutil.copy2(task_rewards_src, data_dir / "task_rewards.py")

    (dest / "environment" / "Dockerfile").write_text(
        _render_harbor_dockerfile(),
        encoding="utf-8",
    )

    for artifact_name in (
        "task.py",
        "manifest.json",
        "__init__.py",
    ):
        src = staging_dir / artifact_name
        if src.exists():
            shutil.copy2(src, data_dir / artifact_name)

    shutil.copy2(
        _explicit_runner_source_path(),
        data_dir / _EXPLICIT_RUNNER_NAME,
    )

    # Install the helper used to submit predictions to the host verifier.
    submit_helper_src = _harbor_submit_source_path()
    if submit_helper_src.exists():
        shutil.copy2(submit_helper_src, data_dir / _HARBOR_SUBMIT_NAME)

    dataset = (
        phase_artifacts.datasets[0]
        if phase_artifacts is not None and phase_artifacts.datasets
        else None
    )

    if loader_ready:
        # Split loader output, then install the deterministic runtime loader.
        _materialise_split_artifacts(
            staging_dir=staging_dir,
            dataset=dataset,
            datasets_root=datasets_root,
            curated_datasets_root=curated_datasets_root,
            agent_data_dir=data_dir,
            eval_data_dir=eval_data_dir,
            task_id=task_id,
            installed_manifest_path=data_dir / "manifest.json",
        )
    else:
        # Models without held-out targets use raw payloads and any staged loader.
        load_src = staging_dir / "load.py"
        if load_src.exists():
            shutil.copy2(load_src, data_dir / "load.py")
        _copy_dataset_payloads_to_environment(
            dataset,
            target_data_dir=data_dir,
            datasets_root=datasets_root,
            curated_datasets_root=curated_datasets_root,
        )

    log_detail(
        logger,
        "%s install task_id=%s to=%s loader_ready=%s",
        PHASE_TAG,
        task_id,
        dest,
        loader_ready,
    )
    try:
        rel_dest = Path(dest).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        rel_dest = Path(dest).as_posix()
    log_stage(
        logger,
        "%s   install -> %s",
        PHASE_TAG,
        rel_dest,
    )
    return dest


def _materialise_split_artifacts(
    *,
    staging_dir: Path,
    dataset: Any | None,
    datasets_root: Path,
    curated_datasets_root: Path,
    agent_data_dir: Path,
    eval_data_dir: Path,
    task_id: str,
    installed_manifest_path: Path | None = None,
) -> None:
    """Materialize agent/verifier HDF5 files from the approved loader.

    The installed loader is then replaced with the deterministic runtime
    loader, and the manifest receives the slicer's observed contract. Failures
    abort installation.
    """
    import importlib.util as _il_util
    import sys as _sys

    from openfinai_pipeline.benchmark.install.runtime_loader import (
        TRIVIAL_RUNTIME_LOADER_SRC,
    )
    from openfinai_pipeline.benchmark.install.slicer import (
        SliceShapeError,
        slice_b_shape,
        write_dataset_h5,
        write_test_gt_h5,
    )

    staged_loader = staging_dir / "load.py"
    if not staged_loader.exists():
        raise FileNotFoundError(
            f"loader_ready=True but {staged_loader} is missing — "
            "phase4 loader generation should have produced it."
        )

    # Resolve the source payload without copying it.
    dataset_data_dir = _resolve_staged_dataset_dir(
        dataset=dataset,
        datasets_root=datasets_root,
        curated_datasets_root=curated_datasets_root,
    )
    if dataset_data_dir is None:
        raise FileNotFoundError(
            f"could not locate staged dataset payload dir for task {task_id}; "
            "phase 2 output may be missing."
        )

    mod_name = f"_install_loader_{abs(hash(str(staged_loader)))}"
    spec = _il_util.spec_from_file_location(mod_name, str(staged_loader))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {staged_loader}")
    module = _il_util.module_from_spec(spec)
    _sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    loader_fn = getattr(module, "load", None)
    if not callable(loader_fn):
        raise ImportError(
            f"{staged_loader} does not export a callable load(data_dir); "
            "regenerate the loader."
        )
    full_output = loader_fn(dataset_data_dir)

    try:
        slice_result = slice_b_shape(full_output)
    except SliceShapeError as exc:
        raise RuntimeError(
            f"install slicer rejected loader output for task {task_id}: {exc}"
        ) from exc

    agent_data_dir.mkdir(parents=True, exist_ok=True)
    eval_data_dir.mkdir(parents=True, exist_ok=True)

    write_dataset_h5(agent_data_dir / "dataset.h5", slice_result)
    write_test_gt_h5(eval_data_dir / "test_ground_truth.h5", slice_result.held_out)

    # Runtime loading reads dataset.h5 and hides held-out test ground truth.
    (agent_data_dir / "load.py").write_text(
        TRIVIAL_RUNTIME_LOADER_SRC, encoding="utf-8"
    )

    log_detail(
        logger,
        "%s slicer materialised task_id=%s shape=%s policy=%s "
        "agent_h5=%s eval_h5=%s",
        PHASE_TAG,
        task_id,
        slice_result.shape,
        slice_result.split_policy,
        agent_data_dir / "dataset.h5",
        eval_data_dir / "test_ground_truth.h5",
    )
    logger.info(
        "%s   slicer: shape=%s policy=%s",
        PHASE_TAG,
        slice_result.shape,
        slice_result.split_policy,
    )

    # Replace staging placeholders with the slicer's observed policy. Only a
    # B-shape bundle has verifier-only test ground truth.
    if installed_manifest_path is not None and installed_manifest_path.exists():
        try:
            existing = json.loads(
                installed_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict):
            existing["split_policy"] = slice_result.split_policy
            existing["has_held_out_test_gt"] = (
                slice_result.shape == "b_shape"
            )
            installed_manifest_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )


def _resolve_staged_dataset_dir(
    *,
    dataset: Any | None,
    datasets_root: Path,
    curated_datasets_root: Path,
) -> Path | None:
    """Find the payload directory read by the generated loader."""
    if dataset is None or not dataset.source_paths:
        return None
    try:
        first_path = Path(dataset.source_paths[0]).resolve()
    except Exception:  # noqa: BLE001 — best-effort
        return None
    for parent in [first_path, *first_path.parents]:
        if parent.name == "data" and parent.is_dir():
            return parent
    return None


# Validation loop


def run_validation_loop(
    candidate: TaskCandidate,
    initial_code: GeneratedCode,
    sandbox: Sandbox,
    staging_dir: Path,
    llm: LLMService,
    paper_context: str,
    base_task_source: str,
    example_task_source: str,
    task_generation_rounds: int = 3,
    log_path: Path | None = None,
    phase_artifacts: PhaseArtifacts | None = None,
    template_reference: str = "",
    interaction_model: str = "gym",
    data_dir: Path | None = None,
    resolved_reward_names: list[str] | None = None,
    task_rewards_path: Path | None = None,
    expected_outputs: list[str] | None = None,
    split_policy: str = "",
    has_held_out_test_gt: bool = True,
) -> ImplementationResult:
    """Retry static checks, pytest, and an end-to-end agent smoke test.

    Failures feed the next generation round. Smoke validation is skipped when
    no dataset directory was resolved.
    """
    code = initial_code
    error_log: list[str] = []
    last_failure_stage: str | None = None

    for attempt in range(1, task_generation_rounds + 1):
        # Preserve the candidate before validation.
        write_staging(staging_dir, code)

        violations = static_check_code(code, sandbox)
        if violations:
            msg = f"[attempt {attempt}] Static violations:\n" + "\n".join(violations)
            log_detail(logger, "%s %s", PHASE_TAG, msg)
            log_stage(
                logger,
                "%s   task round %d/%d -> rejected: static violations (%d): %s",
                PHASE_TAG,
                attempt,
                task_generation_rounds,
                len(violations),
                truncate_oneline(violations[0]),
            )
            error_log.append(msg)
            _log_attempt(log_path, candidate.task_id, attempt, False, msg)
            last_failure_stage = "static"

            if attempt < task_generation_rounds:
                code = generate_task_code(
                    candidate,
                    llm,
                    paper_context,
                    base_task_source,
                    example_task_source,
                    previous_code=code,
                    error_output="\n".join(violations),
                    phase_artifacts=phase_artifacts,
                    template_reference=template_reference,
                    interaction_model=interaction_model,
                    resolved_reward_names=resolved_reward_names,
                    task_rewards_path=task_rewards_path,
                    expected_outputs=expected_outputs,
                    split_policy=split_policy,
                    has_held_out_test_gt=has_held_out_test_gt,
                )
            continue

        passed, test_output = sandbox.run_tests(staging_dir, task_id=candidate.task_id)
        _log_attempt(log_path, candidate.task_id, attempt, passed, test_output)

        if not passed:
            msg = f"[attempt {attempt}] Tests failed:\n{test_output}"
            log_detail(logger, "%s %s", PHASE_TAG, msg)
            log_stage(
                logger,
                "%s   task round %d/%d -> rejected: %s",
                PHASE_TAG,
                attempt,
                task_generation_rounds,
                _short_reject_reason_task(test_output),
            )
            error_log.append(msg)
            last_failure_stage = "pytest"

            if attempt < task_generation_rounds:
                code = generate_task_code(
                    candidate,
                    llm,
                    paper_context,
                    base_task_source,
                    example_task_source,
                    previous_code=code,
                    error_output=test_output,
                    phase_artifacts=phase_artifacts,
                    template_reference=template_reference,
                    interaction_model=interaction_model,
                    resolved_reward_names=resolved_reward_names,
                    task_rewards_path=task_rewards_path,
                    expected_outputs=expected_outputs,
                    split_policy=split_policy,
                    has_held_out_test_gt=has_held_out_test_gt,
                )
            continue

        # Pytest covers structure; smoke validation covers runtime wiring.
        smoke_result = _run_agent_smoke_gate(
            staging_dir=staging_dir,
            data_dir=data_dir,
            interaction_model=interaction_model,
            task_id=candidate.task_id,
        )
        if smoke_result.get("ok", False):
            log_stage(
                logger,
                "%s   task round %d/%d -> approved",
                PHASE_TAG,
                attempt,
                task_generation_rounds,
            )
            return ImplementationResult(
                task_id=candidate.task_id,
                success=True,
                module_path=f"{candidate.scope_id}.{candidate.task_id}",
                attempts=attempt,
                error_log=error_log,
                failure_stage=None,
            )

        # Feed smoke failures into another round or the final sidecar.
        smoke_feedback = format_smoke_failure_for_llm(smoke_result)
        msg = f"[attempt {attempt}] Agent smoke gate failed:\n{smoke_feedback}"
        log_detail(logger, "%s %s", PHASE_TAG, msg)
        log_stage(
            logger,
            "%s   task round %d/%d -> rejected: %s",
            PHASE_TAG,
            attempt,
            task_generation_rounds,
            _short_reject_reason_smoke(smoke_result),
        )
        error_log.append(msg)
        _log_attempt(log_path, candidate.task_id, attempt, False, msg)
        last_failure_stage = "agent_smoke"

        if attempt == task_generation_rounds:
            # Preserve the final structured failure for inspection.
            try:
                write_agent_smoke_failure_sidecar(
                    staging_dir,
                    scope_id=candidate.scope_id,
                    task_id=candidate.task_id,
                    result=smoke_result,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort
                log_detail(
                    logger,
                    "%s agent_smoke sidecar write failed task_id=%s error=%s",
                    PHASE_TAG,
                    candidate.task_id,
                    exc,
                    level=logging.WARNING,
                )

        if attempt < task_generation_rounds:
            code = generate_task_code(
                candidate,
                llm,
                paper_context,
                base_task_source,
                example_task_source,
                previous_code=code,
                error_output=smoke_feedback,
                phase_artifacts=phase_artifacts,
                template_reference=template_reference,
                interaction_model=interaction_model,
                resolved_reward_names=resolved_reward_names,
                task_rewards_path=task_rewards_path,
                expected_outputs=expected_outputs,
                split_policy=split_policy,
                has_held_out_test_gt=has_held_out_test_gt,
            )

    return ImplementationResult(
        task_id=candidate.task_id,
        success=False,
        attempts=task_generation_rounds,
        error_log=error_log,
        failure_stage=last_failure_stage,
    )


def _run_agent_smoke_gate(
    *,
    staging_dir: Path,
    data_dir: Path | None,
    interaction_model: str,
    task_id: str,
) -> dict:
    """Run agent smoke validation and always return its result shape."""
    if data_dir is None:
        # Upstream failures may leave no payload to smoke-test.
        log_detail(
            logger,
            "%s skip agent_smoke task_id=%s: data_dir is None",
            PHASE_TAG,
            task_id,
        )
        return {
            "ran": False,
            "ok": True,  # fail-open per smoke contract
            "errors": ["agent_smoke_skipped:no_data_dir"],
            "modes": {},
            "smoke_dir": "",
            "interaction_model": interaction_model,
        }
    try:
        return agent_smoke_validate(
            staging_dir=staging_dir,
            data_dir=data_dir,
            interaction_model=interaction_model,
        )
    except Exception as exc:  # noqa: BLE001 — defensive, never crash run
        return {
            "ran": False,
            "ok": False,
            "errors": [f"agent_smoke_raised:{type(exc).__name__}:{exc}"],
            "modes": {},
            "smoke_dir": "",
            "interaction_model": interaction_model,
        }


def _short_reject_reason_smoke(smoke_result: dict) -> str:
    """Return the first smoke-mode or setup failure."""
    modes = smoke_result.get("modes") or {}
    for mode_name in ("agent_view", "direct"):
        mode = modes.get(mode_name)
        if isinstance(mode, dict) and not mode.get("ok", False):
            err = mode.get("error") or "unknown"
            return truncate_oneline(f"smoke {mode_name}: {err}")
    errors = smoke_result.get("errors") or []
    if errors:
        return truncate_oneline(f"smoke pre-mode: {errors[0]}")
    return "smoke failed"


def _short_reject_reason_task(test_output: str) -> str:
    """One-line reason for a failed task pytest round.

    Prefers the first ``FAILED ...`` line from pytest output (terse,
    test-name-bearing); falls back to the first non-empty line.
    """
    text = (test_output or "").strip()
    if not text:
        return "tests failed"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("FAILED", "ERROR", "E   ")):
            return truncate_oneline(stripped)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return truncate_oneline(stripped)
    return "tests failed"


# Top-level orchestrator


def implement_task(
    candidate: TaskCandidate,
    llm: LLMService,
    sandbox: Sandbox,
    staging_root: Path,
    tasks_root: Path,
    datasets_root: Path,
    curated_datasets_root: Path,
    log_path: Path | None = None,
    task_generation_rounds: int = 3,
    phase_artifacts: PhaseArtifacts | None = None,
    resolved_reward_names: list[str] | None = None,
    pdf_downloader: PDFDownloader | None = None,
    loader_ready: bool = False,
    has_held_out_test_gt: bool = True,
    ground_truth_provenance: dict[str, Any] | None = None,
    installed_task_ids: set[str] | None = None,
) -> ImplementationResult:
    """Generate, validate, and install one task.

    ``loader_ready`` enables HDF5 slicing; ``has_held_out_test_gt`` controls
    whether instructions describe a verifier-only target.
    """
    log_detail(
        logger,
        "%s implement task_id=%s title=%s",
        PHASE_TAG,
        candidate.task_id,
        candidate.title,
    )

    # Reject ambiguous models before template and test selection.
    interaction_model = _infer_interaction_model(candidate)
    if interaction_model not in ("forecasting", "generative"):
        msg = (
            f"unsupported interaction_model={interaction_model!r} "
            "(auto-pipeline only handles forecasting + generative; "
            "trading/realtime/gym are out of scope)"
        )
        log_detail(
            logger,
            "%s skip task_id=%s %s",
            PHASE_TAG,
            candidate.task_id,
            msg,
            level=logging.WARNING,
        )
        return ImplementationResult(
            task_id=candidate.task_id,
            success=False,
            attempts=0,
            error_log=[f"unsupported_interaction_model={interaction_model}"],
        )

    paper_context = _read_paper_context(candidate, pdf_downloader=pdf_downloader)
    base_task_source = _read_base_task_interface()
    example_task_source = _read_example_task(tasks_root, interaction_model=interaction_model)
    template_reference = _render_template_reference(
        candidate,
        phase_artifacts,
        resolved_reward_names=resolved_reward_names,
        interaction_model=interaction_model,
    )

    # Build generation context from available phase artifacts.
    enriched_chars: int = 0
    if phase_artifacts is not None:
        enriched_task_info = _build_enriched_task_info(
            candidate,
            phase_artifacts,
            resolved_reward_names=list(resolved_reward_names or []),
        )
        enriched_chars = len(enriched_task_info)
    logger.info(
        "%s   implement: interaction=%s enriched=%s",
        PHASE_TAG,
        interaction_model,
        fmt_chars(enriched_chars),
    )

    staging_dir = staging_root / _staging_folder_name(
        candidate.task_id, candidate.ml_task_summary
    )
    write_deterministic_harbor_assets(
        staging_dir,
        candidate,
        phase_artifacts,
        resolved_reward_names or [],
        interaction_model,
        ground_truth_provenance=ground_truth_provenance,
    )

    # Use the assembled reward bundle to render metric contracts.
    task_rewards_path = staging_dir / "task_rewards.py"
    if not task_rewards_path.exists():
        task_rewards_path = None  # type: ignore[assignment]
    expected_outputs: list[str]
    if interaction_model == "forecasting":
        expected_outputs = ["predictions.npy", "predictions.csv"]
    elif interaction_model == "generative":
        expected_outputs = [
            "samples.npy",
            "samples.csv",
            "predictions.npy",
            "predictions.csv",
        ]
    else:
        expected_outputs = []

    try:
        initial_code = generate_task_code(
            candidate,
            llm,
            paper_context,
            base_task_source,
            example_task_source,
            phase_artifacts=phase_artifacts,
            template_reference=template_reference,
            interaction_model=interaction_model,
            resolved_reward_names=list(resolved_reward_names or []),
            task_rewards_path=task_rewards_path,
            expected_outputs=expected_outputs,
            split_policy="",  # known only after slicer; manifest patches this in
            has_held_out_test_gt=has_held_out_test_gt,
        )
    except Exception as exc:
        log_detail(
            logger,
            "%s code_generation_failed task_id=%s error=%s",
            PHASE_TAG,
            candidate.task_id,
            exc,
            level=logging.ERROR,
        )
        return ImplementationResult(
            task_id=candidate.task_id,
            success=False,
            attempts=0,
            error_log=[f"code_generation_failed: {exc}"],
        )

    # Preserve the pre-assembled evaluator across staging writes.
    pre_assembled_eval = staging_dir / "evaluator.py"
    if pre_assembled_eval.exists():
        log_detail(
            logger,
            "%s pre-assembled evaluator found at %s",
            PHASE_TAG,
            pre_assembled_eval,
        )
    else:
        logger.warning(
            "%s no pre-assembled evaluator at %s -- LLM fallback if available",
            PHASE_TAG,
            staging_dir,
        )

    # Resolve the payload used to build the smoke-test layout.
    primary_dataset = (
        phase_artifacts.datasets[0]
        if phase_artifacts is not None and phase_artifacts.datasets
        else None
    )
    dataset_data_dir = _resolve_staged_dataset_dir(
        dataset=primary_dataset,
        datasets_root=datasets_root,
        curated_datasets_root=curated_datasets_root,
    )

    result = run_validation_loop(
        candidate,
        initial_code,
        sandbox,
        staging_dir,
        llm,
        paper_context,
        base_task_source,
        example_task_source,
        task_generation_rounds=task_generation_rounds,
        log_path=log_path,
        phase_artifacts=phase_artifacts,
        template_reference=template_reference,
        interaction_model=interaction_model,
        data_dir=dataset_data_dir,
        resolved_reward_names=list(resolved_reward_names or []),
        task_rewards_path=task_rewards_path,
        expected_outputs=expected_outputs,
        split_policy="",
        has_held_out_test_gt=has_held_out_test_gt,
    )

    if result.success:
        install_task(
            candidate.scope_id,
            candidate.task_id,
            staging_dir,
            tasks_root,
            phase_artifacts=phase_artifacts,
            datasets_root=datasets_root,
            curated_datasets_root=curated_datasets_root,
            loader_ready=loader_ready,
            pdf_path=candidate.pdf_path,
            installed_task_ids=installed_task_ids,
        )

    return result


# Context helpers


def _read_paper_context(
    candidate: TaskCandidate,
    *,
    pdf_downloader: PDFDownloader | None = None,
) -> str:
    """Read capped paper text without low-value sections, or use the abstract."""
    if candidate.pdf_path:
        try:
            downloader = pdf_downloader or PDFDownloader(DownloadConfig())
            page_texts = downloader.extract_page_texts_from_path(
                candidate.pdf_path,
                max_pages=20,
            )
            full_text = "\n".join(page_texts).strip() if page_texts else ""
            if len(full_text) > 100:
                return strip_low_value_sections(full_text)[:_PAPER_CONTEXT_MAX_CHARS]
            fallback = downloader.extract_excerpt_from_path(
                candidate.pdf_path,
                max_chars=_PAPER_CONTEXT_MAX_CHARS,
            )
            if fallback and len(fallback.strip()) > 100:
                return strip_low_value_sections(fallback)[:_PAPER_CONTEXT_MAX_CHARS]
        except Exception as exc:
            logger.warning(
                "%s pdf_read_failed task_id=%s error=%s",
                PHASE_TAG,
                candidate.task_id,
                exc,
            )

    return candidate.paper_abstract[:_PAPER_CONTEXT_MAX_CHARS]


def _read_base_task_interface() -> str:
    """Read the BaseTask source code for inclusion in the prompt."""
    base_path = Path(__file__).resolve().parents[1] / "contracts.py"
    try:
        return base_path.read_text(encoding="utf-8")
    except OSError:
        return "(BaseTask source not found)"


def _read_example_task(tasks_root: Path, interaction_model: str = "gym") -> str:
    """Find a matching installed task.py, or return its inline example.

    Both flat and scoped task layouts are supported.
    """
    if tasks_root.exists():
        candidates: list[Path] = []
        for item in sorted(tasks_root.iterdir()):
            if (
                not item.is_dir()
                or item.name.startswith(("_", "."))
                or item.name == "__pycache__"
            ):
                continue
            direct_candidates = _installed_task_source_candidates(item)
            if direct_candidates:
                candidates.extend(direct_candidates)
                continue
            for sub in sorted(item.iterdir()):
                if (
                    not sub.is_dir()
                    or sub.name.startswith(("_", "."))
                    or sub.name == "__pycache__"
                ):
                    continue
                candidates.extend(_installed_task_source_candidates(sub))

        if interaction_model in {"forecasting", "generative", "trading"}:
            class_name = {
                "forecasting": "ForecastingTask",
                "generative": "GenerativeTask",
                "trading": "TradingTask",
            }[interaction_model]
            for task_py in candidates:
                try:
                    source = task_py.read_text(encoding="utf-8")
                except OSError:
                    continue
                if class_name in source:
                    return source[:20000]
        elif candidates:
            try:
                return candidates[0].read_text(encoding="utf-8")[:20000]
            except OSError:
                pass

    if interaction_model == "forecasting":
        return _INLINE_EXAMPLE_FORECASTING_TASK
    if interaction_model == "generative":
        return _INLINE_EXAMPLE_GENERATIVE_TASK
    return _INLINE_EXAMPLE_TASK


def _render_template_reference(
    candidate: TaskCandidate,
    phase_artifacts: PhaseArtifacts | None = None,
    resolved_reward_names: list[str] | None = None,
    *,
    interaction_model: str = "gym",
) -> str:
    """Render task.py with placeholders as an implementation guide.

    Evaluator and instruction rendering remain deterministic and separate.
    """
    del resolved_reward_names  # retained for API stability; not used
    templates_dir = _TEMPLATES_DIR
    if not templates_dir.is_dir():
        logger.warning("Templates directory not found: %s", templates_dir)
        return ""

    # Derive the canonical task class name.
    raw = candidate.task_id.replace("-", "_")
    class_name = "".join(word.capitalize() for word in raw.split("_"))
    if not class_name:
        class_name = "GeneratedTask"

    data_requirements = [d.get("name", "unnamed") for d in candidate.datasets]

    # Describe payload schemas from Phase 2 hints.
    load_data_docstring = "Load and cache the dataset."
    if phase_artifacts and phase_artifacts.datasets:
        payloads = [
            path
            for dataset in phase_artifacts.datasets
            for path in dataset.payload_files
            if path
        ]
        if payloads:
            load_data_docstring = "Source payload files (schema reference only — runtime sees /data/dataset.h5):\n" + "\n".join(
                f"- {path}" for path in payloads
            )

    task_module_path = "task"
    evaluator_module_path = "evaluator"

    context = {
        "title": candidate.title,
        "task_id": candidate.task_id,
        "class_name": class_name,
        "description": candidate.ml_task_summary[:500],
        "version": "0.1.0",
        "source_papers": [candidate.paper_id] if candidate.paper_id else [],
        "tags": [candidate.scope_id] if candidate.scope_id else [],
        "data_requirements": data_requirements,
        "task_module_path": task_module_path,
        "evaluator_module_path": evaluator_module_path,
        "interaction_model": interaction_model,
        "load_data_docstring": load_data_docstring,
        # load_data is fixed; these behavioral hooks remain LLM-authored.
        "difficulty": "medium",
        "observation_space_impl": '# <<< LLM: return {"shape": ..., "dtype": ...} >>>',
        "action_space_impl": '# <<< LLM: return {"type": "continuous"|"discrete", ...} >>>',
        "reset_impl": "# <<< LLM: return initial observation >>>",
        "step_impl": "# <<< LLM: implement step logic, return (obs, reward, done, info) >>>",
        "extra_rewards_impl": "# <<< LLM: add task-specific rewards if needed >>>",
        "sample_predictions": "[0.1, -0.05, 0.02]",
        "sample_ground_truth": "[0.12, -0.03, 0.01]",
        # Test features are visible; test ground truth is verifier-only.
        "get_features_docstring": "Return the test features the agent predicts on.",
        "get_features_impl": (
            'return self._data["test"]["features"]  # <<< LLM: adapt if post-processing is needed >>>'
        ),
        # Conditional generators keep their test reference hidden.
        "get_reference_docstring": (
            "Return the reference samples. Raises PermissionError when the "
            "task has a held-out test reference (conditional generative)."
        ),
        "get_reference_impl": (
            'if "reference" in self._data:\n'
            '            return self._data["reference"]\n'
            '        raise PermissionError(\n'
            '            "test reference is held out from the agent; "\n'
            '            "use get_train_reference_data() to fit a generator"\n'
            '        )  # <<< LLM: only override when truly unconditional >>>'
        ),
        "get_conditioning_docstring": "Return optional conditioning inputs, or None for unconditional generation.",
        "get_conditioning_impl": (
            'if "train" in self._data:\n'
            '            return self._data["train"].get("features")\n'
            '        return None  # unconditional case'
        ),
        # Trading-only hooks.
        "market_observation_impl": "# <<< LLM: return the current market observation >>>",
        "execute_trade_impl": "# <<< LLM: execute the action and return (reward, info) >>>",
    }

    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        keep_trailing_newline=True,
    )

    # Only task.py is an LLM implementation reference.
    try:
        tmpl = env.get_template("task.py.j2")
        rendered = tmpl.render(**context)
        return f"--- task.py (template reference) ---\n{rendered}"
    except Exception as exc:
        logger.warning("Failed to render task.py.j2: %s", exc)
        return ""


def _log_attempt(
    log_path: Path | None,
    task_id: str,
    attempt: int,
    passed: bool,
    output_snippet: str,
) -> None:
    """Append a JSONL entry to the construction log."""
    if log_path is None:
        return
    entry = {
        "task_id": task_id,
        "attempt": attempt,
        "success": passed,
        "output_snippet": output_snippet,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("%s failed_to_write_construction_log error=%s", PHASE_TAG, exc)
