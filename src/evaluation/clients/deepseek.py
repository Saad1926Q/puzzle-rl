"""DeepSeek Chat Completions client."""

from __future__ import annotations

from typing import Any

from evaluation.clients.common import ChatCompletionAgent, validate_reasoning_effort
from evaluation.constants import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_THINKING,
)


class DeepSeekAgent(ChatCompletionAgent):
    """DeepSeek configuration for the shared Chat Completions adapter."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        thinking: bool = DEFAULT_THINKING,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
        validate_reasoning_effort(reasoning_effort)
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        super().__init__(
            client=client,
            model=model,
            max_tokens=max_tokens,
            provider="DeepSeek",
            truncated_reasons={"length"},
            request_options=self._request_options,
        )

    def _request_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "tool_choice": (
                "auto"
                if self.thinking
                else {"type": "function", "function": {"name": "slide_tile"}}
            ),
            "extra_body": {
                "thinking": {"type": "enabled" if self.thinking else "disabled"}
            },
        }
        if self.thinking:
            options["reasoning_effort"] = self.reasoning_effort
        return options
