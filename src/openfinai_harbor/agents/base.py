"""Shared Harbor chat and prompt scaffold for OpenFinGym agents.

The base composes caller domain prompts with subclass mode rules, constructs
LiteLLM or CLI chats, and transfers usage and rollout details into the agent
context. Subclasses own execution and response parsing; their mode prompt must
stay aligned with that parser.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.llms.chat import Chat
from harbor.llms.lite_llm import LiteLLM
from harbor.models.agent.context import AgentContext

from openfinai_harbor.llms.claude_cli import ClaudeCLIChat, ClaudeCLIConfig


# Fail-fast checks; LiteLLM still performs its own lookup.
_PROVIDER_ENV_VAR = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Routes through the Claude Code CLI subprocess (subscription auth)
# instead of LiteLLM; anything after the slash is forwarded verbatim
# to ``claude --model``.
_CLAUDE_CLI_PREFIX = "claude-cli/"


class BaseFinAgent(BaseAgent):
    """Shared base for OpenFinGym agents (single-shot, ReAct, future)."""

    SUPPORTS_ATIF: bool = False
    SUPPORTS_WINDOWS: bool = False

    # Appended to the caller's domain prompt; subclass's response parser
    # depends on this text — do NOT let callers override via kwarg.
    MODE_PROMPT_ADDENDUM: str = ""

    # Neutral fallback for shell-script callers that can't easily pass
    # multi-line system prompts.
    DEFAULT_DOMAIN_PROMPT: str = (
        "You are a careful data scientist solving a quantitative benchmark "
        "task. Follow the task instruction precisely."
    )

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        temperature: float | None = None,
        system_prompt: str | None = None,
        system_prompt_path: str | Path | None = None,
        api_base: str | None = None,
        session_id: str | None = None,
        collect_rollout_details: bool = False,
        max_thinking_tokens: int | None = None,
        reasoning_effort: str | None = None,
        model_info: dict | None = None,
        llm_kwargs: dict | None = None,
        llm_call_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Forward extras (e.g. legacy ``provider``); harbor.BaseAgent
        # accepts *args/**kwargs and ignores unknowns.
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

        if not model_name:
            raise ValueError(
                "model_name is required; pass a provider-prefixed identifier "
                "like 'openrouter/openai/gpt-4o-mini' or 'hosted_vllm/Qwen3-0.6B'."
            )

        resolved_prompt = self._load_system_prompt(
            system_prompt, system_prompt_path
        )
        self._temperature = temperature
        self._system_prompt = self._compose_system_prompt(resolved_prompt)
        self._collect_rollout_details = bool(collect_rollout_details)
        self._llm_call_kwargs = dict(llm_call_kwargs or {})

        # claude-cli/ backbone: subprocess-driven CLI under subscription
        # auth. No API key, no LiteLLM; _make_chat returns ClaudeCLIChat.
        if model_name.startswith(_CLAUDE_CLI_PREFIX):
            self._llm = None
            extra_args: tuple[str, ...] = ()
            if reasoning_effort:
                extra_args = ("--effort", str(reasoning_effort))
            self._claude_cli_config = ClaudeCLIConfig(
                model=model_name[len(_CLAUDE_CLI_PREFIX) :],
                system_prompt=self._system_prompt,
                extra_args=extra_args,
            )
            return

        self._claude_cli_config = None

        # Fail-fast on missing provider env var; LiteLLM would otherwise
        # surface this as a deep AuthenticationError after its retry loop.
        self._verify_api_key_present(model_name)

        # Per Terminus2's pattern: fold temperature into llm_kwargs so
        # LiteLLM forwards it into every acompletion call.
        constructor_kwargs = dict(llm_kwargs or {})
        if temperature is not None:
            constructor_kwargs["temperature"] = temperature

        self._llm = LiteLLM(
            model_name=model_name,
            api_base=api_base,
            collect_rollout_details=collect_rollout_details,
            session_id=session_id,
            max_thinking_tokens=max_thinking_tokens,
            reasoning_effort=reasoning_effort,
            model_info=model_info,
            **constructor_kwargs,
        )

    # system prompt composition

    @staticmethod
    def _load_system_prompt(
        inline_text: str | None, path: str | Path | None
    ) -> str | None:
        """Pick the system prompt source — inline string or file at ``path``.

        Relative ``path`` values are resolved against the current working
        directory at agent construction (matches SkyRL's data-path
        convention). ``BaseFinAgent.__init__`` runs on the *host* — Harbor
        instantiates the agent class before handing it an
        ``environment``, so the host filesystem is what gets read.
        """
        if inline_text is not None and path is not None:
            raise ValueError(
                "Pass either system_prompt or system_prompt_path, not both."
            )
        if path is None:
            return inline_text
        prompt_path = Path(path)
        try:
            return prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"system_prompt_path={path!r} not found "
                f"(resolved to {prompt_path.resolve()}). "
                "Ensure the launcher's cwd is the OpenFinGym repo root."
            ) from exc

    def _compose_system_prompt(self, caller_prompt: str | None) -> str:
        """Combine caller's domain prompt with the subclass mode addendum.

        Strip-and-join with two newlines so subclass MODE_PROMPT_ADDENDUM
        can carry leading/trailing whitespace without accumulating blank
        lines.
        """
        domain = (caller_prompt or self.DEFAULT_DOMAIN_PROMPT).rstrip()
        if not self.MODE_PROMPT_ADDENDUM:
            return domain
        return f"{domain}\n\n{self.MODE_PROMPT_ADDENDUM.lstrip()}"

    # chat / context helpers

    def _make_chat(self) -> Chat | ClaudeCLIChat:
        """Return a fresh chat object seeded with the composed system prompt.

        For LiteLLM backbones: harbor's ``Chat`` has no constructor
        kwarg for the system prompt, so we seed via direct mutation
        of ``_messages`` (the supported pattern, also used by
        ``Chat.reset_response_chain``'s sibling flows).

        For the ``claude-cli/`` backbone: the system prompt is already
        baked into the ``ClaudeCLIConfig`` (passed to ``claude -p`` as
        ``--system-prompt`` on every call), so ``_messages`` starts
        empty — the subscription session has no separate "system" turn.
        """
        if self._claude_cli_config is not None:
            return ClaudeCLIChat(self._claude_cli_config)
        chat = Chat(self._llm)
        chat._messages.append(
            {"role": "system", "content": self._system_prompt}
        )
        return chat

    @staticmethod
    def _apply_chat_to_context(
        chat: Chat, context: AgentContext, n_episodes: int
    ) -> None:
        """Mirror chat-derived totals + rollout_details + n_episodes onto context.

        Token / cost fields stay at their ``None`` defaults if the chat
        recorded zero usage (i.e. no successful LLM call). This matches the
        existing ``_apply_usage_totals`` semantics and preserves a clear
        signal for trials that crashed before producing any output.
        """
        rollout_details = chat.rollout_details
        if rollout_details:
            context.rollout_details = rollout_details

        existing_metadata = context.metadata or {}
        context.metadata = {**existing_metadata, "n_episodes": int(n_episodes)}

        if chat.total_input_tokens or chat.total_output_tokens:
            context.n_input_tokens = chat.total_input_tokens
            context.n_output_tokens = chat.total_output_tokens
        if chat.total_cache_tokens:
            context.n_cache_tokens = chat.total_cache_tokens
        if chat.total_cost:
            context.cost_usd = float(chat.total_cost)

    # private helpers

    @staticmethod
    def _verify_api_key_present(model_name: str) -> None:
        """Raise a named ``RuntimeError`` if the provider's env var is missing.

        Only checks providers we explicitly know about. ``hosted_vllm/...``
        and other prefixes (anthropic-bedrock, vertex_ai, etc.) are skipped
        — LiteLLM handles them, and a missing key surfaces from there.
        """
        prefix, _, _ = model_name.partition("/")
        env_var = _PROVIDER_ENV_VAR.get(prefix.lower())
        if env_var and not os.environ.get(env_var):
            raise RuntimeError(
                f"{env_var} is not set. Set it in your environment before "
                f"invoking a BaseFinAgent with model_name={model_name!r}."
            )
