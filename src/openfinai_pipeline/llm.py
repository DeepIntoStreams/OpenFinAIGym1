import logging
from typing import Any

from langchain.agents.structured_output import ProviderStrategy
from langchain.tools import BaseTool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from openfinai_pipeline.settings import LLMConfig
from langchain.agents.middleware import ToolCallLimitMiddleware

from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


providers = {
    "openai": ChatOpenAI,
    "openrouter": ChatOpenRouter,
    "anthropic": ChatAnthropic,
    "bedrock": ChatBedrock,
    "ollama": ChatOllama,
}


class LLMService:

    def __init__(self, cfg: LLMConfig) -> None:
        self.provider = cfg.provider
        self.model = cfg.providers[cfg.provider].model
        llm_type = providers[self.provider]
        timeout_sec = cfg.providers[cfg.provider].timeout_sec
        # ChatOpenRouter's `timeout` kwarg maps to the openrouter SDK's
        # `timeout_ms` (milliseconds). Every other provider in `providers`
        # treats `timeout` as seconds. Convert once at the boundary so the
        # rest of the pipeline can keep using seconds uniformly.
        timeout_kwarg: float | int = (
            int(timeout_sec * 1000) if self.provider == "openrouter" else timeout_sec
        )
        self.llm = llm_type(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            model=self.model,
            timeout=timeout_kwarg,
            max_retries=cfg.retry.max_retries,
        )
        self.max_tool_calls = cfg.tool_max_turns

    def complete_structured(
        self, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        llm = self.llm.with_structured_output(schema)
        payload = llm.invoke(prompt)
        payload["provider"] = self.provider
        payload["model"] = self.model
        return payload

    def complete_structured_with_tools(
        self,
        prompt: str,
        schema: dict[str, Any],
        tools: list[BaseTool],
    ) -> dict[str, Any]:
        agent = create_agent(
            self.llm,
            tools,
            response_format=ProviderStrategy(schema),
            middleware=[ToolCallLimitMiddleware(run_limit=self.max_tool_calls)],
        )
        result = agent.invoke({"messages": [HumanMessage(prompt)]})
        payload = result["structured_response"]
        payload["provider"] = self.provider
        payload["model"] = self.model
        return payload

    def complete_or_fallback(
        self,
        prompt: str,
        schema: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return self.complete_structured(prompt, schema)
        except Exception as e:
            logger.warning("LLM call failed, using fallback: %s", e)
            out = dict(fallback)
            if isinstance(out.get("issues"), list):
                out["issues"].append(str(e))
            return out
