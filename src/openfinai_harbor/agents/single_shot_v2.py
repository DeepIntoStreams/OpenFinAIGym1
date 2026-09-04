"""Generate, execute, and iteratively repair a ``train.py`` task solver.

Held-out targets remain available only to the verifier. The fenced-code
prompt contract is defined here because the response parser depends on it.
"""
from __future__ import annotations

import json
import re
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent as ATIFAgent,
    FinalMetrics,
    Step,
    ToolCall,
    Trajectory,
)

from .base import BaseFinAgent


# Response-parser contract appended by BaseFinAgent._compose_system_prompt.

_SINGLE_SHOT_MODE_ADDENDUM = textwrap.dedent(
    """\
    OUTPUT FORMAT
    -------------
    Write a SINGLE, self-contained Python script (``train.py``) that
    solves the task exactly as the instruction below prescribes
    (data loading, deliverable, and submission are all specified there).

    Wrap the script in one fenced markdown block and output ONLY that
    block — no surrounding prose::

        ```python
        # your code here
        ```
    """
)


_DEFAULT_USER_TEMPLATE = """\
TASK INSTRUCTION (verbatim from instruction.md):
================================================
{instruction}

TASK CLASS INTERFACE (head of task.py for reference):
=====================================================
{task_py_head}

Now write the Python script that solves this task.
"""


_PYTHON_CODE_PATTERNS = (
    re.compile(r"```python\n(.*?)```", re.DOTALL),
    re.compile(r"```\n(.*?)```", re.DOTALL),
)

_MAX_DEBUG_ATTEMPTS = 5
_EXEC_TIMEOUT_SEC = 3600
_EXEC_MAX_STDOUT = 50000
_EXEC_MAX_STDERR = 10000


class SingleShotLLMAgent(BaseFinAgent):
    """Generate a solver and repair it from local execution feedback."""

    MODE_PROMPT_ADDENDUM = _SINGLE_SHOT_MODE_ADDENDUM

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        max_tokens: int = 8000,
        user_prompt_template: str | None = None,
        task_py_head_lines: int = 80,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._max_tokens = int(max_tokens)
        self._user_prompt_template = (
            user_prompt_template or _DEFAULT_USER_TEMPLATE
        )
        self._task_py_head_lines = int(task_py_head_lines)

    # BaseAgent interface

    @staticmethod
    def name() -> str:
        return "simple-llm"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Collect best-effort runtime context for the initial prompt."""
        try:
            from openfinai_harbor.agents._env_probe import collect_env_probe
            self._env_probe_text = await collect_env_probe(environment)
        except Exception as exc:
            self.logger.warning("env probe failed: %s", exc)
            self._env_probe_text = ""

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Generate, validate, and execute a solver, retrying with feedback."""
        # Best-effort fetch of /data/task.py head for prompt context.
        task_py_head = await self._fetch_task_py_head(environment)

        self.logger.info(
            "calling LLM model=%s max_tokens=%d temperature=%s",
            self.model_name,
            self._max_tokens,
            self._temperature,
        )
        user_text = self._user_prompt_template.format(
            instruction=instruction, task_py_head=task_py_head
        )
        # Empty when setup could not collect the probe.
        probe_text = getattr(self, "_env_probe_text", "") or ""
        if probe_text:
            user_text = (
                "ENVIRONMENT PROBE (recorded during setup)\n"
                + "-" * 41 + "\n"
                + probe_text
                + "\n\n"
                + user_text
            )

        # Detect which harbor_submit variant the instruction requires.
        required_submit = _detect_submit_function(instruction)

        chat = self._make_chat()
        code: str = ""
        raw_response: str = ""
        prompt: str = user_text
        actual_turns: int = 0

        for debug_attempt in range(_MAX_DEBUG_ATTEMPTS + 1):
            self.logger.info(
                "LLM attempt %d / %d", debug_attempt + 1, _MAX_DEBUG_ATTEMPTS + 1
            )
            try:
                response = await chat.chat(prompt, max_tokens=self._max_tokens)
                raw_response = (response.content or "").strip()
                actual_turns += 1
            except Exception as exc:
                self.logger.error(
                    "LLM call failed on attempt %d: %s", debug_attempt + 1, exc
                )
                break

            # /logs/agent is bind-mounted to trial_paths.agent_dir on the host.
            (self.logs_dir / f"llm_response_{debug_attempt}.txt").write_text(
                raw_response, encoding="utf-8"
            )

            code = _extract_python_block(raw_response)
            if not code:
                self.logger.warning(
                    "No Python code block on attempt %d", debug_attempt + 1
                )
                if debug_attempt < _MAX_DEBUG_ATTEMPTS:
                    prompt = (
                        "Your response contained no Python code block. "
                        "Please provide the complete train.py wrapped in a "
                        "```python\n...\n``` fenced block."
                    )
                continue

            # harbor_submit presence check
            if required_submit and required_submit not in code:
                self.logger.warning(
                    "Attempt %d: code missing required submit call '%s'",
                    debug_attempt + 1,
                    required_submit,
                )
                if debug_attempt < _MAX_DEBUG_ATTEMPTS:
                    prompt = (
                        f"Your train.py is missing a call to `{required_submit}`. "
                        "The task instruction requires you to import and call "
                        f"`{required_submit}` to submit results to the host "
                        "verifier. Please add it and output the complete "
                        "corrected train.py."
                    )
                continue

            # upload and execute
            await environment.exec("mkdir -p /workspace", user="root")
            train_py_path = self.logs_dir / "train.py"
            train_py_path.write_text(code, encoding="utf-8")
            await environment.upload_file(
                source_path=train_py_path,
                target_path="/workspace/train.py",
            )
            self.logger.info(
                "uploaded train.py (%d chars) on attempt %d",
                len(code),
                debug_attempt + 1,
            )

            exec_result = await self._execute_train_py(environment)
            stdout = (exec_result.stdout or "")[:_EXEC_MAX_STDOUT]
            stderr = (exec_result.stderr or "")[:_EXEC_MAX_STDERR]
            exec_output = (
                stdout + ("\n" if stdout and stderr else "") + stderr
            ).strip()
            if exec_result.return_code == 0:
                self.logger.info(
                    "train.py executed successfully on attempt %d\n%s",
                    debug_attempt + 1,
                    exec_output,
                )
                print(
                    f"[attempt {debug_attempt + 1}] train.py succeeded:\n{exec_output}",
                    flush=True,
                )
                break

            # Execution failed — build feedback for next iteration.
            self.logger.warning(
                "train.py failed (rc=%d) on attempt %d\n%s",
                exec_result.return_code,
                debug_attempt + 1,
                exec_output,
            )
            print(
                f"[attempt {debug_attempt + 1}] train.py failed "
                f"(rc={exec_result.return_code}):\n{exec_output}",
                flush=True,
            )
            if debug_attempt < _MAX_DEBUG_ATTEMPTS:
                prompt = (
                    f"Your train.py failed with exit code "
                    f"{exec_result.return_code}. Please fix the error and "
                    "output the complete corrected train.py.\n\n"
                    f"Output:\n{exec_output}"
                )

        # Write the canonical llm_response.txt (last iteration) for tooling.
        (self.logs_dir / "llm_response.txt").write_text(
            raw_response, encoding="utf-8"
        )

        # Persist BEFORE the extraction guard so failed trials still emit
        # trajectory/conversation/answer.json for SFT collection.
        self._persist_agent_artifacts(
            chat=chat,
            raw_response=raw_response,
            extracted_code=code,
            user_text=user_text,
            turns=actual_turns,
        )
        self._apply_chat_to_context(chat, context, n_episodes=1)

        if not code:
            raise RuntimeError(
                "LLM response contained no Python code block after all attempts; "
                f"see {self.logs_dir / 'llm_response.txt'}"
            )

    # persistence

    def _persist_agent_artifacts(
        self,
        *,
        chat: Any,
        raw_response: str,
        extracted_code: str,
        user_text: str,
        turns: int = 1,
    ) -> None:
        """Write conversation.json, answer.json, trajectory.json to logs_dir.

        Called *before* the extraction guard so failed trials (no Python
        block produced) still emit trajectory.json for SFT collection and
        post-hoc analysis. Token / cost stats are read defensively via
        ``getattr`` so this also works when ``chat`` came from a backend
        that doesn't expose every counter.

        Persistence-only system-prompt repair: some chat backends keep
        ``self._system_prompt`` out of ``chat._messages`` because they
        deliver it via a different channel (``claude-cli/`` passes it as
        ``--system-prompt`` on the subprocess command line; see
        ``BaseFinAgent._make_chat``). The LLM still sees the same system
        prompt at inference — only the persistence-side view differs.
        We patch that here so trajectory.json / conversation.json are
        backend-independent and SFT data prep can always read the
        canonical system step. This does NOT change what's sent to the
        LLM (``chat._messages`` is read into a local copy and the
        original list is untouched).
        """
        try:
            messages = list(chat._messages)
        except Exception:
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": raw_response},
            ]

        # Ensure the persisted view starts with the actual system prompt.
        # No-op when the chat backend already seeded ``_messages`` with
        # the system turn (LiteLLM path); restores it when the backend
        # delivered the system prompt out-of-band (claude-cli path).
        system_prompt = getattr(self, "_system_prompt", None)
        if isinstance(system_prompt, str) and system_prompt:
            if not messages or messages[0].get("role") != "system":
                messages = [
                    {"role": "system", "content": system_prompt}
                ] + messages

        try:
            (self.logs_dir / "conversation.json").write_text(
                json.dumps(messages, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.warning("failed to write conversation.json: %s", exc)

        n_in = getattr(chat, "total_input_tokens", None)
        n_out = getattr(chat, "total_output_tokens", None)
        n_cache = getattr(chat, "total_cache_tokens", None)
        cost = getattr(chat, "total_cost", None)

        try:
            (self.logs_dir / "answer.json").write_text(
                json.dumps(
                    {
                        "answer": extracted_code,
                        "raw_response_chars": len(raw_response),
                        "extracted_code_chars": len(extracted_code or ""),
                        "turns": turns,
                        "n_input_tokens": n_in,
                        "n_output_tokens": n_out,
                        "n_cache_tokens": n_cache,
                        "cost_usd": float(cost) if cost else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.warning("failed to write answer.json: %s", exc)

        try:
            atif_trajectory = self._convert_conversation_to_atif(
                messages,
                extracted_code=extracted_code,
                total_input_tokens=int(n_in or 0),
                total_output_tokens=int(n_out or 0),
                cost_usd=float(cost) if cost else None,
            )
            (self.logs_dir / "trajectory.json").write_text(
                json.dumps(
                    atif_trajectory.to_json_dict(),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.warning("failed to write trajectory.json: %s", exc)

    def _convert_conversation_to_atif(
        self,
        conversation: list[dict[str, str]],
        *,
        extracted_code: str,
        total_input_tokens: int,
        total_output_tokens: int,
        cost_usd: float | None = None,
    ) -> Trajectory:
        """Convert the single-shot conversation history to an ATIF Trajectory.

        Mapping rules (subset of ``ReActFinAgent._convert_conversation_to_atif``):

        - ``role: system``    → Step(source="system")
        - ``role: user``      → Step(source="user")
        - ``role: assistant`` → Step(source="agent") with parsed
          ``<reasoning>`` → ``reasoning_content`` and the pre-extracted
          code (passed in by the caller — covers <python> tag /
          markdown fence / bare-body fallbacks) →
          ``tool_calls=[ToolCall(function_name="python_executor",
          arguments={"code": ...})]``.

        Single-shot has exactly one assistant turn, so there is no
        observation step to attach (no smoke run was executed).
        """
        steps: list[Step] = []
        step_id = 1
        now_iso = datetime.now(timezone.utc).isoformat()

        for msg in conversation:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""

            if role == "system":
                steps.append(
                    Step(
                        step_id=step_id,
                        timestamp=now_iso,
                        source="system",
                        message=content,
                    )
                )
                step_id += 1
            elif role == "user":
                steps.append(
                    Step(
                        step_id=step_id,
                        timestamp=now_iso,
                        source="user",
                        message=content,
                    )
                )
                step_id += 1
            elif role == "assistant":
                reasoning_match = re.search(
                    r"<reasoning>(.*?)</reasoning>", content, re.DOTALL
                )
                reasoning = (
                    reasoning_match.group(1).strip()
                    if reasoning_match
                    else None
                )
                tool_calls: list[ToolCall] | None = None
                if extracted_code:
                    tool_calls = [
                        ToolCall(
                            tool_call_id=f"call_{step_id}",
                            function_name="python_executor",
                            arguments={"code": extracted_code},
                        )
                    ]
                steps.append(
                    Step(
                        step_id=step_id,
                        timestamp=now_iso,
                        source="agent",
                        model_name=self.model_name,
                        message=content,
                        reasoning_content=reasoning,
                        tool_calls=tool_calls,
                    )
                )
                step_id += 1

        return Trajectory(
            schema_version="ATIF-v1.6",
            session_id=str(uuid.uuid4()),
            agent=ATIFAgent(
                name=self.name(),
                version=self.version() or "0.2.0",
                model_name=self.model_name,
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=total_input_tokens or None,
                total_completion_tokens=total_output_tokens or None,
                total_cost_usd=cost_usd if cost_usd else None,
                total_steps=len(steps),
            ),
        )

    # helpers

    async def _fetch_task_py_head(
        self, environment: BaseEnvironment
    ) -> str:
        """Best-effort: read the first N lines of ``/data/task.py``."""
        try:
            result = await environment.exec(
                f"head -n {self._task_py_head_lines} /data/task.py",
                timeout_sec=10,
            )
            if result.return_code == 0 and result.stdout:
                return result.stdout
        except Exception as exc:
            self.logger.debug("could not fetch task.py head: %s", exc)
        return "(task.py preview unavailable)"

    async def _execute_train_py(self, environment: BaseEnvironment) -> Any:
        """Stage /data + /workspace/train.py into a temp dir and run it.

        Mirrors the staging approach used by ``ReActAgent._save_candidate_train_py``
        so the script sees all data files in its working directory.
        """
        stage_cmd = (
            "set -e; "
            "rm -rf /workspace/.openfinai/exec_run; "
            "mkdir -p /workspace/.openfinai/exec_run; "
            "if [[ -d /data ]]; then cp -a /data/. /workspace/.openfinai/exec_run/; fi; "
            "cp /workspace/train.py /workspace/.openfinai/exec_run/train.py"
        )
        stage_result = await environment.exec(stage_cmd)
        if stage_result.return_code != 0:
            # Treat staging failure as an execution failure so the caller
            # can feed the error back to the LLM.
            return stage_result
        return await environment.exec(
            "cd /workspace/.openfinai/exec_run && python3 train.py",
            timeout_sec=_EXEC_TIMEOUT_SEC,
        )


def _detect_submit_function(instruction: str) -> str | None:
    """Return the harbor_submit variant the instruction requires, or None.

    Checks for ``harbor_submit_curated`` first (more specific) so that
    instructions containing both strings map to the curated variant.
    """
    if "harbor_submit_curated" in instruction:
        return "harbor_submit_curated"
    if "harbor_submit" in instruction:
        return "harbor_submit"
    return None


def _extract_python_block(text: str) -> str:
    """Pull a ```python …``` (or bare ``` …```) fenced block out of ``text``."""
    for pattern in _PYTHON_CODE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    stripped = text.strip()
    if stripped.startswith(("import ", "from ", "#!")):
        return stripped
    return ""
