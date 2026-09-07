"""Qwen3.5 OpenAI-compatible Chat Completions client."""

from __future__ import annotations

from typing import Any

from evaluation.clients.common import ChatCompletionAgent
from evaluation.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_QWEN_PRESENCE_PENALTY,
    DEFAULT_QWEN_REPETITION_PENALTY,
    DEFAULT_QWEN_TEMPERATURE,
    DEFAULT_QWEN_TOP_K,
    DEFAULT_QWEN_TOP_P,
)


class QwenAgent(ChatCompletionAgent):
    """Qwen3.5 configuration for the shared Chat Completions adapter."""

    def __init__(
        self,
        *,
        api_key: str = "not-required",
        model: str = DEFAULT_QWEN_MODEL,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        thinking: bool = False,
        reasoning_effort: str = "low",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_QWEN_TEMPERATURE,
        top_p: float = DEFAULT_QWEN_TOP_P,
        top_k: int = DEFAULT_QWEN_TOP_K,
        presence_penalty: float = DEFAULT_QWEN_PRESENCE_PENALTY,
        repetition_penalty: float = DEFAULT_QWEN_REPETITION_PENALTY,
        client: Any | None = None,
    ) -> None:
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if reasoning_effort not in {"low", "medium", "xhigh"}:
            raise ValueError("reasoning_effort must be low, medium, or xhigh")
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)

        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        super().__init__(
            client=client,
            model=model,
            max_tokens=max_tokens,
            provider="Qwen",
            truncated_reasons={"length"},
            request_options=self._request_options,
        )

    def _request_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "extra_body": {
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
                "chat_template_kwargs": {"enable_thinking": self.thinking},
            },
        }
        if self.thinking:
            options["reasoning_effort"] = self.reasoning_effort
        return options
