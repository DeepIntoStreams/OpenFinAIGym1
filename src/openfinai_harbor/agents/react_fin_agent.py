"""
ReActFinAgent — a ReAct-pattern data-science agent for Harbor.

Implements multi-turn reasoning-and-acting with stateful code execution
inside the Harbor environment container. The agent calls an LLM via
LiteLLM, parses ``<python>`` / ``<answer>`` tags, and executes code
through a lightweight HTTP code-execution server installed during
``setup()``.

The financial-domain wording previously baked into this agent has been
extracted to ``examples/prompts/financial_expert.txt`` (caller-supplied
via the ``system_prompt`` kwarg). The ``<python>`` / ``<information>`` /
``<answer>`` tag protocol stays here as ``MODE_PROMPT_ADDENDUM`` because
the response parser depends on it — the parser and the prompt are one
contract.
"""
import asyncio
import json
import re
import socket
import tempfile
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment
from harbor.llms.base import ContextLengthExceededError
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent as ATIFAgent,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)

from .base import BaseFinAgent


def _pick_free_port() -> int:
    """Return an unused loopback TCP port for the in-container exec server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

# Stateful code-execution server (uploaded into the container at setup time).
# ``__EXEC_PORT__`` is substituted per trial in ``setup()`` — trials may share
# the host network namespace (Podman ``--network=host``, or the host-network
# compose shim in ``run_trial``), so a fixed port would collide across
# concurrent trials.

_EXEC_SERVER_PY = textwrap.dedent(
    r'''
    """Tiny HTTP server that maintains a persistent Python session."""
    import http.server, json, sys, io, traceback

    _G = {"__builtins__": __builtins__}

    class _H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            code = json.loads(body).get("code", "")
            out, err = io.StringIO(), io.StringIO()
            _old = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = out, err
            try:
                exec(compile(code, "<agent>", "exec"), _G)
            except Exception:
                traceback.print_exc()
            finally:
                sys.stdout, sys.stderr = _old
            resp = json.dumps({"stdout": out.getvalue(), "stderr": err.getvalue()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        def log_message(self, *_a):
            pass

    http.server.HTTPServer(("127.0.0.1", __EXEC_PORT__), _H).serve_forever()
'''
).lstrip()

# Response-parser contract appended by BaseFinAgent._compose_system_prompt.
# Do not override via kwarg — the parser depends on this exact text.

_REACT_MODE_ADDENDUM = textwrap.dedent(
    """\
    OUTPUT FORMAT (ReAct multi-turn)
    --------------------------------
    1. Write executable Python code inside <python>…</python> tags. Each
       block runs in a persistent session — variables survive across
       blocks.
    2. The execution result appears in <information>…</information> tags.
    3. When you have a final answer, wrap it in <answer>…</answer> tags.

    Follow the instruction below for how to load data, what to produce,
    and how to submit.

    RESPONSE TEMPLATE
    -----------------
    <reasoning>Why you are doing this step.</reasoning>
    <python>Your executable Python code.</python>
    <information>Execution output appears here.</information>
    …repeat as needed…
    <reasoning>How you arrived at the final solution.</reasoning>
    <answer>Your final answer or deliverable.</answer>
    """
)

MAX_STDOUT = 5000
MAX_STDERR = 2000


class ReActFinAgent(BaseFinAgent):
    """ReAct-pattern multi-turn agent for data-science tasks on Harbor."""

    MODE_PROMPT_ADDENDUM = _REACT_MODE_ADDENDUM

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        max_turns: int = 15,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.max_turns = int(max_turns)
        self._exec_port = _pick_free_port()

    # BaseAgent interface

    @staticmethod
    def name() -> str:
        return "react-fin-agent"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install deps, start stateful code-execution server, and run env probe."""
        await environment.exec("pip install -q numpy pandas", timeout_sec=120)

        await environment.exec("mkdir -p /workspace/.openfinai")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(_EXEC_SERVER_PY.replace("__EXEC_PORT__", str(self._exec_port)))
            tmp.flush()
            await environment.upload_file(
                tmp.name, "/workspace/.openfinai/exec_server.py"
            )

        await environment.exec(
            "nohup python /workspace/.openfinai/exec_server.py "
            "> /workspace/.openfinai/server.log 2>&1 &"
        )
        await asyncio.sleep(2)

        health = await environment.exec(
            f"curl -sf -X POST http://127.0.0.1:{self._exec_port} "
            "-H 'Content-Type: application/json' "
            """-d '{"code":"print(42)"}'"""
        )
        if health.return_code != 0:
            self.logger.warning("exec_server health-check failed, retrying …")
            await asyncio.sleep(3)

        # Best-effort env probe: stashed on ``self._env_probe_text`` so
        # ``run()`` can prepend it to the first user message. Empty
        # string on failure so a broken probe doesn't sink the trial.
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
        """Run the ReAct loop: think → code → observe → repeat.

        Each turn routes through ``harbor.llms.Chat.chat``, which talks to
        vLLM/OpenRouter/etc. via Harbor's ``LiteLLM`` wrapper. When the
        agent is constructed with ``collect_rollout_details=True`` (SkyRL's
        HarborGenerator sets this), per-turn token IDs + logprobs land in
        ``chat.rollout_details`` and are mirrored onto ``context`` at exit.
        """
        chat = self._make_chat()
        # Prepend the env probe (collected during setup) to the FIRST
        # user message only. Empty when the probe failed at setup time.
        probe_text = getattr(self, "_env_probe_text", "") or ""
        if probe_text:
            next_user_text = (
                "ENVIRONMENT PROBE (recorded during setup)\n"
                + "-" * 41 + "\n"
                + probe_text
                + "\n\n"
                + instruction
            )
        else:
            next_user_text = instruction
        final_answer = ""
        actual_turns = 0

        for turn in range(self.max_turns):
            actual_turns = turn + 1
            self.logger.info("turn %d / %d", actual_turns, self.max_turns)

            try:
                response = await chat.chat(next_user_text)
            except ContextLengthExceededError:
                # Propagate so SkyRL's _harbor_agent_loop classifies as
                # ContextLengthExceededError (reward=0) instead of retrying.
                raise
            except Exception as exc:
                self.logger.error(
                    "LLM call failed on turn %d: %s", actual_turns, exc
                )
                # Chat raised before extending _messages; record the failing
                # prompt manually so the saved conversation reflects what the
                # model saw, then retry with an error observation next turn.
                chat._messages.append(
                    {"role": "user", "content": next_user_text}
                )
                next_user_text = (
                    f"<information>LLM error: {exc}. Try again.</information>"
                )
                continue

            assistant_text = _postprocess(response.content or "")
            # Overwrite chat's raw assistant content with the tag-truncated
            # version so the saved conversation matches what the parser saw.
            chat._messages[-1]["content"] = assistant_text

            if _has_answer(assistant_text):
                final_answer = _extract_answer(assistant_text)
                break

            code = _extract_code(assistant_text)
            if code:
                output = await self._execute_code(environment, code)
                next_user_text = f"\n<information>{output}</information>\n"
            else:
                next_user_text = (
                    "<information>No Python code found. "
                    "Write code in <python>CODE</python> or provide "
                    "<answer>ANSWER</answer>.</information>"
                )

        # persist outputs for the verifier
        await self._save_agent_outputs(
            environment,
            final_answer,
            chat.messages,
            actual_turns,
            total_input_tokens=chat.total_input_tokens or 0,
            total_output_tokens=chat.total_output_tokens or 0,
        )

        # -- populate AgentContext (token totals + rollout_details + n_episodes)
        self._apply_chat_to_context(chat, context, n_episodes=actual_turns)

    # code execution

    async def _execute_code(
        self,
        environment: BaseEnvironment,
        code: str,
    ) -> str:
        """Send *code* to the in-container exec server and return output."""
        payload = json.dumps({"code": code})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write(payload)
            tmp.flush()
            await environment.upload_file(tmp.name, "/tmp/_code_input.json")

        result = await environment.exec(
            f"curl -sf -X POST http://127.0.0.1:{self._exec_port} "
            "-H 'Content-Type: application/json' "
            "-d @/tmp/_code_input.json",
            timeout_sec=300,
        )

        if result.return_code != 0:
            stderr = (result.stderr or "")[:MAX_STDERR]
            return f"Execution error (curl): {stderr}"

        try:
            data = json.loads(result.stdout or "{}")
            stdout = (data.get("stdout") or "")[:MAX_STDOUT]
            stderr = (data.get("stderr") or "")[:MAX_STDERR]
            return (stdout + stderr).strip() or "(no output)"
        except json.JSONDecodeError:
            return (result.stdout or "")[:MAX_STDOUT]

    # ATIF trajectory conversion

    def _convert_conversation_to_atif(
        self,
        conversation: list[dict[str, str]],
        total_input_tokens: int,
        total_output_tokens: int,
        cost_usd: float | None = None,
    ) -> Trajectory:
        """Convert the ReAct conversation history to an ATIF Trajectory.

        Mapping rules:

        - ``role: system``    → Step(source="system")
        - ``role: user`` (first, i.e. the task instruction) →
          Step(source="user")
        - ``role: assistant`` → Step(source="agent") with parsed
          ``<reasoning>`` → ``reasoning_content`` and ``<python>`` →
          ``tool_calls=[ToolCall(function_name="python_executor",
          arguments={"code": ...})]``
        - Subsequent ``role: user`` messages whose body contains an
          ``<information>…</information>`` block → merged into the
          preceding agent step's ``observation.results`` as an
          ``ObservationResult(source_call_id=<matching tool_call_id>,
          content=<obs>)``.
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
                info_match = re.search(
                    r"<information>(.*?)</information>", content, re.DOTALL
                )
                if info_match and steps and steps[-1].source == "agent":
                    obs_content = info_match.group(1).strip()
                    prev = steps[-1]
                    if prev.observation and prev.observation.results:
                        prev.observation.results.append(
                            ObservationResult(content=obs_content)
                        )
                    else:
                        source_id = None
                        if prev.tool_calls:
                            source_id = prev.tool_calls[-1].tool_call_id
                        prev.observation = Observation(
                            results=[
                                ObservationResult(
                                    source_call_id=source_id,
                                    content=obs_content,
                                )
                            ]
                        )
                else:
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
                    reasoning_match.group(1).strip() if reasoning_match else None
                )

                code_match = re.search(
                    r"<python>(.*?)</python>", content, re.DOTALL
                )
                tool_calls: list[ToolCall] | None = None
                if code_match:
                    tc_id = f"call_{step_id}"
                    tool_calls = [
                        ToolCall(
                            tool_call_id=tc_id,
                            function_name="python_executor",
                            arguments={"code": code_match.group(1).strip()},
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
                version=self.version() or "0.1.0",
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

    # persistence

    async def _save_agent_outputs(
        self,
        environment: BaseEnvironment,
        answer: str,
        conversation: list[dict[str, str]],
        turns: int,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        cost_usd: float | None = None,
    ) -> None:
        """Persist answer, conversation, ATIF trajectory, train.py, and model artifacts.

        JSON artifacts (``answer.json`` / ``conversation.json`` /
        ``trajectory.json``) are written **host-direct** to
        ``self.logs_dir`` (the bind-mounted ``/logs/agent`` dir) so
        they pick up the host umask (mode ``0o644``) — the host user
        inspecting the trial can read them without sudo, and the SFT
        pipeline can collect them off the host without going through
        the container. The previous ``environment.upload_file`` path
        produced mode-``0o600`` files unreadable to anyone except the
        container user.

        ``trajectory.json`` is the ATIF-v1.6 form of the conversation,
        emitted via ``_convert_conversation_to_atif`` — that's what
        downstream SFT trajectory-collection consumes.
        """
        host_dir = self.logs_dir
        host_dir.mkdir(parents=True, exist_ok=True)

        # Some backends transmit the system prompt outside ``conversation``.
        # Add it only to the persisted copy so trajectory formats stay uniform;
        # this does not change the messages sent to the model.
        system_prompt = getattr(self, "_system_prompt", None)
        if isinstance(system_prompt, str) and system_prompt:
            if not conversation or conversation[0].get("role") != "system":
                conversation = [
                    {"role": "system", "content": system_prompt}
                ] + list(conversation)

        # 1. answer.json (host-direct)
        try:
            (host_dir / "answer.json").write_text(
                json.dumps(
                    {
                        "answer": answer,
                        "turns": int(turns),
                        "n_input_tokens": int(total_input_tokens),
                        "n_output_tokens": int(total_output_tokens),
                        "cost_usd": cost_usd,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.warning("failed to write answer.json: %s", exc)

        # 2. conversation.json (host-direct)
        try:
            (host_dir / "conversation.json").write_text(
                json.dumps(conversation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.warning("failed to write conversation.json: %s", exc)

        # 3. trajectory.json (host-direct, ATIF-v1.6)
        try:
            atif_trajectory = self._convert_conversation_to_atif(
                conversation,
                total_input_tokens=int(total_input_tokens),
                total_output_tokens=int(total_output_tokens),
                cost_usd=cost_usd,
            )
            (host_dir / "trajectory.json").write_text(
                json.dumps(
                    atif_trajectory.to_json_dict(),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.warning("failed to write trajectory.json: %s", exc)

        # Use cat + host write so train.py lands mode 0o644 (in-container cp
        # would give 0o600, unreadable to the host SFT collector).
        try:
            cat_result = await environment.exec(
                "cat /workspace/train.py 2>/dev/null || true",
                timeout_sec=30,
            )
            train_text = cat_result.stdout or ""
            if train_text.strip():
                (host_dir / "train.py").write_text(train_text, encoding="utf-8")
        except Exception as exc:
            self.logger.warning("failed to write train.py: %s", exc)

        # Save trained artifacts to survive teardown. The final chmod -R
        # a+rwX is load-bearing: in-container cp lands mode 0o600 so the
        # host user can't rm -rf the trial dir without it.
        try:
            agent_dir = "/logs/agent"
            await environment.exec(f"mkdir -p {agent_dir}/model")
            for search_dir in ("/workspace", "/data", "/tmp/openfinai_eval_*"):
                await environment.exec(
                    f"find {search_dir} -maxdepth 3 "
                    r"\( -name '*.pt' -o -name '*.pth' -o -name '*.pkl' "
                    r"-o -name 'losses_history.csv' -o -name 'final_results.csv' "
                    r"-o -name 'test_score.txt' \) "
                    f"-exec cp -v --parents {{}} {agent_dir}/model/ \\; "
                    "2>/dev/null || true"
                )
            await environment.exec(
                f"chmod -R a+rwX {agent_dir}/model 2>/dev/null || true"
            )
        except Exception as exc:
            self.logger.warning("failed to collect model artifacts: %s", exc)


def _postprocess(text: str) -> str:
    """Truncate after the first closing action tag."""
    for tag in ("</python>", "</answer>"):
        if tag in text:
            return text[: text.index(tag) + len(tag)]
    return text


def _has_answer(text: str) -> bool:
    return bool(text and "<answer>" in text and "</answer>" in text)


def _extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_code(text: str) -> str | None:
    if not text or "<python>" not in text or "</python>" not in text:
        return None
    m = re.search(r"<python>(.*?)</python>", text, re.DOTALL)
    if not m:
        return None
    code = m.group(1)
    inner = re.search(r"```python(.*?)```", code, re.DOTALL)
    if inner:
        return inner.group(1).strip()
    return code.strip() or None
