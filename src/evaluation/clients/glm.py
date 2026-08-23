"""Z.AI GLM Chat Completions client."""

from __future__ import annotations

from typing import Any

from evaluation.clients.common import api_error_metadata, parse_chat_response
from evaluation.constants import (
    DEFAULT_GLM_BASE_URL,
    DEFAULT_GLM_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_THINKING,
    SLIDE_TILE_TOOL,
)
from evaluation.protocol import build_messages
from puzzle3.board import Board


class GLMAgent:
    """One-request-per-state GLM-4.5-Air adapter."""

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
        self.client = client
        self.model = model
        self.thinking = thinking
        # Kept for consistent CLI metadata; GLM-4.5-Air controls thinking with
        # thinking.type rather than a per-level reasoning_effort parameter.
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.last_response_metadata: dict[str, Any] = {}

    def next_action(self, board: Board) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(board),
            "tools": [SLIDE_TILE_TOOL],
            # Z.AI currently supports only auto tool selection.
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
            "extra_body": {
                "thinking": {
                    "type": "enabled" if self.thinking else "disabled",
                }
            },
        }
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            self.last_response_metadata = api_error_metadata(exc)
            raise
        try:
            result, self.last_response_metadata = parse_chat_response(
                response,
                provider="GLM",
                truncated_reasons={"length", "model_context_window_exceeded"},
            )
        except RuntimeError:
            self.last_response_metadata = {
                "status": "invalid_response",
                "error": "response did not contain a completion choice",
            }
            raise
        return result
