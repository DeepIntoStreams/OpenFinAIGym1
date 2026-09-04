"""Deterministic, no-LLM agent for end-to-end Harbor smoke tests.

Family-specific trivial solvers exercise task shapes and verifier transport.
The agent subclasses Harbor directly to avoid LLM setup and emits one ATIF
step so trajectory consumers can process the run.
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent as ATIFAgent,
    FinalMetrics,
    Step,
    ToolCall,
    Trajectory,
)


# Baseline source-code templates.

_FORECASTING_TEMPLATE = '''\
"""DeterministicBaselineAgent — last-value persistence forecasting baseline."""
import os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/data")
from task import {class_name}

EXPECTED_SHAPE = {expected_shape!r}


def fit_model(task, config):
    return None


def {secondary}(task, model, config):
    feats = task.get_features()
    if hasattr(feats, "shape") and len(EXPECTED_SHAPE) == 3 and feats.ndim == 3:
        N, T, C = feats.shape
        H = EXPECTED_SHAPE[1]
        out_C = EXPECTED_SHAPE[2]
        last = feats[:, -1:, :]
        if C == out_C:
            return np.repeat(last, H, axis=1).astype(np.float32)
    return np.zeros(EXPECTED_SHAPE, dtype=np.float32)


def run():
    task = {class_name}()
    task.load_data()
    model = fit_model(task, {{}})
    preds = {secondary}(task, model, {{}})
    preds = np.asarray(preds, dtype=np.float32)
    if preds.shape != EXPECTED_SHAPE:
        preds = np.zeros(EXPECTED_SHAPE, dtype=np.float32)
    if os.environ.get("VERIFIER_URL", "").strip():
        from harbor_submit import submit_and_persist
        submit_and_persist(predictions=preds)
    else:
        np.save("predictions.npy", preds)


if __name__ == "__main__":
    run()
'''


_GENERATION_TEMPLATE = '''\
"""DeterministicBaselineAgent — Gaussian-noise generation baseline."""
import os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/data")
from task import {class_name}

EXPECTED_SHAPE = {expected_shape!r}


def fit_model(task, config):
    return None


def sample_model(task, model, n_samples, config):
    rng = np.random.default_rng(seed=0)
    return rng.normal(0.0, 0.01, size=EXPECTED_SHAPE).astype(np.float32)


def run():
    task = {class_name}()
    task.load_data()
    model = fit_model(task, {{}})
    samples = sample_model(task, model, EXPECTED_SHAPE[0], {{}})
    samples = np.asarray(samples, dtype=np.float32)
    if samples.shape != EXPECTED_SHAPE:
        samples = np.zeros(EXPECTED_SHAPE, dtype=np.float32)
    if os.environ.get("VERIFIER_URL", "").strip():
        from harbor_submit import submit_and_persist
        submit_and_persist(predictions=samples)
    else:
        np.save("samples.npy", samples)


if __name__ == "__main__":
    run()
'''


_YIELDCURVE_TEMPLATE = '''\
"""DeterministicBaselineAgent — yieldcurve zero-curve baseline."""
import os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/data")
from task import {class_name}, EXPECTED_CURVE_SIZE


def _infer_n_test(task):
    feats = task.get_features()
    if isinstance(feats, dict) and "contexts" in feats:
        return int(np.asarray(feats["contexts"]).shape[0])
    if hasattr(feats, "shape") and feats.shape:
        return int(feats.shape[0])
    return 100


def run():
    task = {class_name}()
    task.load_data()
    N = _infer_n_test(task)
    base = np.zeros((N, EXPECTED_CURVE_SIZE), dtype=np.float32)
    preds = np.stack([base, base - 0.05, base + 0.05], axis=1).astype(np.float32)
    if os.environ.get("VERIFIER_URL", "").strip():
        from harbor_submit import submit_and_persist
        submit_and_persist(predictions=preds)
    else:
        np.save("predictions.npy", preds)


if __name__ == "__main__":
    run()
'''


# Static-analysis helpers (same shape detection as full_audit.py audit 5).


_FRAMEWORK_BASE_NAMES = ("ForecastingTask", "GenerativeTask")


def _detect_framework_class(task_py_text: str) -> str | None:
    """Return the framework class name from a task.py source string.

    Filters out the in-file fallback shims (``TaskMetadata`` /
    ``ForecastingTask`` / ``GenerativeTask``) and picks the user class
    that inherits from one of the framework bases.
    """
    import ast

    try:
        tree = ast.parse(task_py_text)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name: str | None = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if base_name in _FRAMEWORK_BASE_NAMES:
                return node.name
    # Fallback — first concrete class that isn't a known shim.
    skip = {"TaskMetadata", "ForecastingTask", "GenerativeTask"}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name not in skip:
            return node.name
    return None


def _resolve_shape_const(task_py_text: str, candidates: list[str]) -> tuple | None:
    """Execute task.py in a stubbed namespace + look up a tuple-valued constant."""
    ns: dict[str, Any] = {"__name__": "_baseline_probe"}
    try:
        import numpy as _np

        ns["np"] = _np
    except Exception:  # pragma: no cover — numpy is always present at runtime
        pass

    class _StubBase:
        def __init__(self, *a, **kw): ...

        def __init_subclass__(cls, **kw): ...

    ns.setdefault("ForecastingTask", _StubBase)
    ns.setdefault("GenerativeTask", _StubBase)
    ns.setdefault("TaskMetadata", _StubBase)
    try:
        exec(compile(task_py_text, "<baseline-probe>", "exec"), ns)
    except Exception:
        # Regex fallback — pull bare tuple literals out of the source.
        decls: dict[str, str] = {}
        for m in re.finditer(
            r"^([A-Z][A-Z0-9_]*)\s*=\s*([^\n]+)$", task_py_text, re.MULTILINE
        ):
            decls[m.group(1)] = m.group(2).split("#", 1)[0].strip().rstrip(",")
        for name in candidates:
            if name not in decls:
                continue
            seen: set[str] = set()
            rhs = decls[name]
            while rhs in decls and rhs not in seen:
                seen.add(rhs)
                rhs = decls[rhs]
            try:
                val = eval(rhs, {"__builtins__": {}}, {})
                if isinstance(val, tuple):
                    return val
            except Exception:
                continue
        return None
    for name in candidates:
        val = ns.get(name)
        if isinstance(val, tuple):
            return val
    return None


def _is_yieldcurve(task_py_text: str) -> bool:
    """The yieldcurve baseline has its own (N, 3, K) stack contract."""
    return "EXPECTED_CURVE_SIZE" in task_py_text


def _detect_secondary_fn(train_py_text: str) -> str:
    """Forecasting tasks use predict_model; generation tasks use sample_model."""
    if re.search(r"^def predict_model\b", train_py_text, re.MULTILINE):
        return "predict_model"
    if re.search(r"^def sample_model\b", train_py_text, re.MULTILINE):
        return "sample_model"
    return "predict_model"


def compose_baseline(
    task_py_text: str,
    train_py_text: str,
    is_forecasting: bool,
) -> str:
    """Build the baseline train.py source for a given task.

    Picks the family-appropriate template + resolves the task's expected
    output shape from its module-level constants. Raises ``RuntimeError``
    if the task class or shape can't be resolved (better to fail fast
    than ship an empty/zero file silently).
    """
    cls = _detect_framework_class(task_py_text)
    if cls is None:
        raise RuntimeError(
            "DeterministicBaselineAgent: could not detect framework class in task.py"
        )

    if _is_yieldcurve(task_py_text):
        return _YIELDCURVE_TEMPLATE.format(class_name=cls)

    if is_forecasting:
        secondary = _detect_secondary_fn(train_py_text)
        shape = _resolve_shape_const(
            task_py_text,
            ["EXPECTED_TEST_PREDICTION_SHAPE", "EXPECTED_TEST_Y_SHAPE"],
        )
        if shape is None:
            raise RuntimeError(
                "DeterministicBaselineAgent: no EXPECTED_TEST_PREDICTION_SHAPE "
                "found in task.py for forecasting baseline."
            )
        return _FORECASTING_TEMPLATE.format(
            class_name=cls, secondary=secondary, expected_shape=shape
        )

    shape = _resolve_shape_const(
        task_py_text,
        ["EXPECTED_SAMPLE_SHAPE", "EXPECTED_SAMPLES_SHAPE"],
    )
    if shape is None:
        raise RuntimeError(
            "DeterministicBaselineAgent: no EXPECTED_SAMPLE_SHAPE found in "
            "task.py for generation baseline."
        )
    return _GENERATION_TEMPLATE.format(class_name=cls, expected_shape=shape)


class DeterministicBaselineAgent(BaseAgent):
    """No-LLM agent that uploads a family-specific baseline train.py.

    Inherits directly from harbor's ``BaseAgent`` (not ``BaseFinAgent``)
    because there is no LLM to construct and no API key to validate.
    Accepts an ignored ``model_name`` so the standard
    ``openfinai_harbor.run_trial`` invocation (which always passes one)
    still works.
    """

    SUPPORTS_ATIF: bool = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        # Swallow framework kwargs (temperature, max_turns, …); this agent
        # ignores them. Only harbor.BaseAgent-known kwargs are forwarded.
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            logger=logger,
            mcp_servers=kwargs.get("mcp_servers"),
            skills_dir=kwargs.get("skills_dir"),
        )

    # BaseAgent interface

    @staticmethod
    def name() -> str:
        return "deterministic-baseline"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Ensure /workspace + the .openfinai staging dir exist."""
        await environment.exec("mkdir -p /workspace/.openfinai", user="root")
        # Pre-copy so the verifier sees a starter file if upload below errors.
        await environment.exec(
            "cp /data/train.py /workspace/train.py 2>/dev/null || true",
            timeout_sec=15,
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Detect task family → upload the matching baseline → emit ATIF."""
        task_py = await self._read_container_file(environment, "/data/task.py")
        train_py = await self._read_container_file(environment, "/data/train.py")
        if task_py is None or train_py is None:
            raise RuntimeError(
                "DeterministicBaselineAgent: /data/task.py or /data/train.py is "
                "missing inside the container; cannot compose baseline."
            )

        # Structural decision: predict_model => forecasting; sample_model
        # => generation. Matches full_audit.py.
        is_forecasting = (
            re.search(r"^def predict_model\b", train_py, re.MULTILINE) is not None
        )

        baseline_code = compose_baseline(
            task_py_text=task_py,
            train_py_text=train_py,
            is_forecasting=is_forecasting,
        )

        # Host-side compile so broken templates fail NOW, not after the
        # verifier runs and complains.
        compile(baseline_code, "<deterministic-baseline>", "exec")

        await self._write_train_py(environment, baseline_code)

        # One synthetic agent step with the baseline as a tool-call.
        self._persist_trajectory(baseline_code)

        # Token counts / cost stay None: this agent makes no LLM calls.
        existing_meta = context.metadata or {}
        context.metadata = {**existing_meta, "n_episodes": 1, "no_llm": True}

    # helpers

    async def _read_container_file(
        self, environment: BaseEnvironment, path: str
    ) -> str | None:
        check = await environment.exec(f"test -f {path}", timeout_sec=10)
        if check.return_code != 0:
            return None
        result = await environment.exec(f"cat {path}", timeout_sec=30)
        if result.return_code != 0:
            return None
        return result.stdout or None

    async def _write_train_py(
        self, environment: BaseEnvironment, code: str
    ) -> None:
        """Upload + atomic-mv to /workspace/train.py (matches Single-Shot path)."""
        candidate = "/workspace/.openfinai/candidate_train.py"
        train_path = "/workspace/train.py"
        await environment.exec("mkdir -p /workspace/.openfinai", timeout_sec=15)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(code)
            tmp.flush()
            await environment.upload_file(tmp.name, candidate)

        # In-container py_compile guards against host/container Python drift.
        compile_result = await environment.exec(
            f"python -m py_compile {candidate}", timeout_sec=30
        )
        if compile_result.return_code != 0:
            err = (compile_result.stderr or compile_result.stdout or "")[:4000]
            raise RuntimeError(
                "DeterministicBaselineAgent: in-container py_compile of the "
                f"baseline failed:\n{err}"
            )

        promote = await environment.exec(
            f"mv -f {candidate} {train_path}", timeout_sec=15
        )
        if promote.return_code != 0:
            err = (promote.stderr or promote.stdout or "")[:4000]
            raise RuntimeError(
                f"DeterministicBaselineAgent: failed to mv candidate to {train_path}:\n{err}"
            )

        # Mirror to host logs_dir for offline inspection.
        try:
            (self.logs_dir / "train.py").write_text(code, encoding="utf-8")
        except OSError:
            pass

    def _persist_trajectory(self, baseline_code: str) -> None:
        """Write the minimal ATIF trajectory.json (single agent step)."""
        now_iso = datetime.now(timezone.utc).isoformat()
        step = Step(
            step_id=1,
            timestamp=now_iso,
            source="agent",
            model_name=self.model_name or "deterministic",
            message="DeterministicBaselineAgent: uploaded family-appropriate baseline.",
            tool_calls=[
                ToolCall(
                    tool_call_id="call_1",
                    function_name="python_executor",
                    arguments={"code": baseline_code},
                )
            ],
        )
        traj = Trajectory(
            schema_version="ATIF-v1.6",
            session_id=str(uuid.uuid4()),
            agent=ATIFAgent(
                name=self.name(),
                version=self.version() or "0.1.0",
                model_name=self.model_name,
            ),
            steps=[step],
            final_metrics=FinalMetrics(total_steps=1),
        )
        try:
            (self.logs_dir / "trajectory.json").write_text(
                json.dumps(traj.to_json_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
