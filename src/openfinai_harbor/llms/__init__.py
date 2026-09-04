"""Local LLM backbones for openfinai_harbor agents.

Harbor's stock ``LiteLLM`` covers API-billed providers (OpenRouter,
Anthropic, OpenAI, hosted vLLM). ``claude_cli`` adds a subprocess
backbone for the Claude Code CLI (``claude -p``) so agents can use the
user's Claude Code subscription instead of an API key.
"""
from .claude_cli import ClaudeCLIChat, ClaudeCLIConfig


__all__ = ["ClaudeCLIChat", "ClaudeCLIConfig"]
