"""CrofAI OpenAI-compatible Chat Completions client."""

from __future__ import annotations

from typing import Any, Sequence

from evaluation.clients.common import (
    api_error_metadata,
    parse_chat_response,
    validate_reasoning_effort,
)
from evaluation.constants import (
    DEFAULT_CROF_BASE_URL,
    DEFAULT_CROF_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_THINKING,
    SLIDE_TILE_TOOL,
)
from evaluation.protocol import HistoryTurn, build_chat_completion_messages
from puzzle3.board import Board


class CrofAgent:
    """One-request-per-state CrofAI adapter using Chat Completions."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_CROF_MODEL,
        base_url: str = DEFAULT_CROF_BASE_URL,
        thinking: bool = DEFAULT_THINKING,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
        validate_reasoning_effort(reasoning_effort)
        self.client = client
        self.model = model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.last_response_metadata: dict[str, Any] = {}

    def next_action(
        self,
        board: Board,
        history: Sequence[HistoryTurn] = (),
        *,
        include_reasoning: bool = False,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": build_chat_completion_messages(
                board, history, include_reasoning=include_reasoning
            ),
            "tools": [SLIDE_TILE_TOOL],
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
        }
        if self.thinking:
            request["reasoning_effort"] = self.reasoning_effort
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            self.last_response_metadata = api_error_metadata(exc)
            raise
        try:
            result, self.last_response_metadata = parse_chat_response(
                response,
                provider="CrofAI",
                truncated_reasons={"length"},
            )
        except RuntimeError:
            self.last_response_metadata = {
                "status": "invalid_response",
                "error": "response did not contain a completion choice",
            }
            raise
        return result
