"""Z.AI GLM Chat Completions client."""

from __future__ import annotations

from typing import Any

from evaluation.clients.common import ChatCompletionAgent
from evaluation.constants import (
    DEFAULT_GLM_BASE_URL,
    DEFAULT_GLM_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_THINKING,
)


class GLMAgent(ChatCompletionAgent):
    """GLM configuration for the shared Chat Completions adapter."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_GLM_MODEL,
        base_url: str = DEFAULT_GLM_BASE_URL,
        thinking: bool = DEFAULT_THINKING,
        reasoning_effort: str = "medium",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
        self.thinking = thinking
        # Kept for consistent CLI metadata; GLM controls thinking with
        # ``thinking.type`` rather than a per-level reasoning parameter.
        self.reasoning_effort = reasoning_effort
        super().__init__(
            client=client,
            model=model,
            max_tokens=max_tokens,
            provider="GLM",
            truncated_reasons={"length", "model_context_window_exceeded"},
            request_options=self._request_options,
        )

    def _request_options(self) -> dict[str, Any]:
        return {
            "tool_choice": "auto",
            "extra_body": {
                "thinking": {"type": "enabled" if self.thinking else "disabled"}
            },
        }
