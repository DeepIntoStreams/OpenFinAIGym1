"""Adapt ``claude -p`` to the Harbor chat interface.

Calls do not use ``--resume``; each one serializes the full message history so
retries and caller-side message mutations retain normal chat semantics.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harbor.llms.base import LLMResponse


_LOG = logging.getLogger(__name__)


# The environment override accommodates slow CLI calls.
def _resolve_default_timeout_sec() -> float:
    raw = os.environ.get("CLAUDE_CLI_TIMEOUT_SEC")
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return 1800.0


@dataclass
class ClaudeCLIConfig:
    """Static config for the CLI shim, built once in ``BaseFinAgent.__init__``."""

    model: str
    """Value forwarded to ``claude --model``. CLI accepts aliases
    (``opus``/``sonnet``/``haiku``) or full IDs (``claude-opus-4-7``)."""

    system_prompt: str
    """Replaces Claude Code's default system prompt via ``--system-prompt``.
    Critical: this also bypasses the CC default-prompt cache-creation
    overhead, so the agent prompt is the *only* system content."""

    binary: str = "claude"
    """Path or name of the CLI. ``shutil.which`` resolves it at first use."""

    timeout_sec: float = field(
        default_factory=lambda: _resolve_default_timeout_sec()
    )
    """Per-call subprocess timeout. Default 1800s (30 min); CLI cold
    start + extended thinking can run several minutes. Override via the
    ``CLAUDE_CLI_TIMEOUT_SEC`` env var (read at config-construction
    time) without touching code."""

    extra_args: tuple[str, ...] = field(default_factory=tuple)
    """Escape hatch for power users — appended verbatim to argv."""


# Internal cache token estimate from ``claude -p`` JSON usage. The CLI
# returns separate ``cache_creation_input_tokens`` and
# ``cache_read_input_tokens``; ``harbor.llms.Chat`` collapses both into a
# single ``cache_tokens`` total, so we mirror that.
def _sum_cache_tokens(usage: dict[str, Any]) -> int:
    return int(usage.get("cache_creation_input_tokens", 0) or 0) + int(
        usage.get("cache_read_input_tokens", 0) or 0
    )


def _serialize_history(messages: list[dict[str, str]], new_user_text: str) -> str:
    """Flatten conversation history + the next user turn into a single prompt.

    ``claude -p`` accepts one prompt argument; multi-turn ReAct loops
    therefore replay the whole transcript on every turn. The leading
    ``role:`` markers make it unambiguous to the model that the trailing
    block is the latest user turn awaiting a response.

    System messages are skipped here — they go to ``--system-prompt``.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        content = (msg.get("content") or "").rstrip()
        if not content:
            continue
        label = "USER" if role == "user" else "ASSISTANT"
        parts.append(f"{label}:\n{content}")
    parts.append(f"USER:\n{new_user_text.rstrip()}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


class ClaudeCLIChat:
    """Duck-type substitute for ``harbor.llms.Chat`` driven by ``claude -p``.

    Implements the exact subset that ``BaseFinAgent`` /
    ``SingleShotLLMAgent`` / ``ReActFinAgent`` actually touch:

    * ``_messages`` / ``messages`` — mutable conversation list
    * ``await chat.chat(prompt, **kwargs)`` — returns ``LLMResponse``
    * ``total_{input,output,cache}_tokens`` / ``total_cost`` properties
    * ``rollout_details`` — always empty list (no RL training surface here)
    * ``reset_response_chain()`` — no-op (no Responses-API chain to reset)
    """

    def __init__(self, config: ClaudeCLIConfig) -> None:
        self._config = config
        self._messages: list[dict[str, str]] = []
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0
        self._cumulative_cache_tokens = 0
        # ``total_cost`` stays 0 on this backbone — billing is the
        # user's Claude Code subscription, not pay-per-token. The CLI's
        # own ``total_cost_usd`` is kept for diagnostics in
        # ``total_virtual_cost`` (API-equivalent dollars).
        self._cumulative_virtual_cost = 0.0
        self._virtual_cost_usd_last = 0.0

    # harbor.llms.Chat surface

    @property
    def messages(self) -> list[dict[str, str]]:
        return self._messages

    @property
    def total_input_tokens(self) -> int:
        return self._cumulative_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._cumulative_output_tokens

    @property
    def total_cache_tokens(self) -> int:
        return self._cumulative_cache_tokens

    @property
    def total_cost(self) -> float:
        # Subscription billing — no real-dollar cost to report. See
        # ``total_virtual_cost`` for the CLI's API-equivalent figure.
        return 0.0

    @property
    def total_virtual_cost(self) -> float:
        """API-equivalent dollars the CLI would have billed if this had
        gone through the metered API. Diagnostic only; not propagated
        into ``AgentContext.cost_usd``."""
        return self._cumulative_virtual_cost

    @property
    def rollout_details(self) -> list:
        return []

    def reset_response_chain(self) -> None:
        """No-op. Each ``claude -p`` call is its own session, no chain to reset."""

    async def chat(self, prompt: str, **_kwargs: Any) -> LLMResponse:
        """Invoke ``claude -p`` with the conversation so far + ``prompt``.

        On success: append both turns to ``_messages`` and accumulate
        usage — matching ``harbor.llms.Chat.chat`` semantics so callers
        like ``ReActFinAgent``'s error-recovery branch keep working
        unchanged.

        ``**_kwargs`` (e.g. ``max_tokens`` from ``SingleShotLLMAgent``)
        is intentionally swallowed: ``claude -p`` exposes no equivalent
        knob, and silently ignoring is friendlier than raising on a
        config the caller can't easily strip.
        """
        full_prompt = _serialize_history(self._messages, prompt)
        stdout = await self._invoke_cli(full_prompt)
        payload = _parse_cli_json(stdout)
        if payload.get("is_error"):
            err = payload.get("result") or payload.get("api_error_status") or "?"
            raise RuntimeError(f"claude -p returned an error: {err}")

        content = (payload.get("result") or "").rstrip()
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cache_tokens = _sum_cache_tokens(usage)
        # The CLI reports API-equivalent cost, not subscription billing. Keep it
        # diagnostic-only so downstream cash totals remain zero.
        self._virtual_cost_usd_last = float(
            payload.get("total_cost_usd", 0.0) or 0.0
        )
        self._cumulative_virtual_cost += self._virtual_cost_usd_last

        self._cumulative_input_tokens += input_tokens
        self._cumulative_output_tokens += output_tokens
        self._cumulative_cache_tokens += cache_tokens

        _LOG.info(
            "claude-cli turn done: in=%d out=%d cache=%d "
            "virtual_cost_usd=$%.4f (subscription — no real billing)",
            input_tokens,
            output_tokens,
            cache_tokens,
            self._virtual_cost_usd_last,
        )

        self._messages.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": content},
            ]
        )
        return LLMResponse(content=content, model_name=self._config.model)

    # subprocess plumbing

    async def _invoke_cli(self, prompt: str) -> str:
        binary = shutil.which(self._config.binary) or self._config.binary
        argv = [
            binary,
            "-p",
            prompt,
            "--model",
            self._config.model,
            "--system-prompt",
            self._config.system_prompt,
            "--output-format",
            "json",
            "--no-session-persistence",
            "--disable-slash-commands",
            # Pure-LLM provider: forbid every built-in tool. The CLI's
            # documented sentinel for "disable all tools" is --tools "".
            "--tools",
            "",
            *self._config.extra_args,
        ]
        _LOG.info(
            "claude-cli invoke: model=%s prompt_chars=%d history_msgs=%d",
            self._config.model,
            len(prompt),
            len(self._messages),
        )

        # Use a clean tempdir as cwd so the host project's CLAUDE.md and
        # other auto-discovered context don't leak into the agent's
        # session. ``claude -p`` honors cwd for CLAUDE.md auto-loading.
        with tempfile.TemporaryDirectory(prefix="claude_cli_") as cwd:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_subprocess_env(),
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=self._config.timeout_sec
                )
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise TimeoutError(
                    f"claude -p timed out after {self._config.timeout_sec}s"
                ) from exc

        if proc.returncode != 0:
            stderr = (stderr_b or b"").decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(
                f"claude -p exited {proc.returncode}: {stderr.strip() or '(no stderr)'}"
            )
        return (stdout_b or b"").decode("utf-8", errors="replace")


def _parse_cli_json(stdout: str) -> dict[str, Any]:
    """Parse the JSON document ``claude -p --output-format json`` prints.

    The CLI prints exactly one JSON object on stdout; defensive in case
    a future version interleaves whitespace or banner text.
    """
    stdout = stdout.strip()
    if not stdout:
        raise RuntimeError("claude -p produced empty stdout")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(
                f"claude -p stdout was not JSON: {stdout[:500]!r}"
            ) from None
        return json.loads(stdout[start : end + 1])


def _subprocess_env() -> dict[str, str]:
    """Environment to pass to ``claude``. Strip vars that would force
    API-key auth and bypass the user's logged-in subscription."""
    env = dict(os.environ)
    # ANTHROPIC_API_KEY in env makes the CLI prefer API billing over
    # the subscription OAuth token — the whole point of this backbone
    # is the subscription, so drop it for the child process.
    env.pop("ANTHROPIC_API_KEY", None)
    return env
