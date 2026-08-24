"""Qwen3.5 OpenAI-compatible Chat Completions client."""

from __future__ import annotations

from typing import Any

from evaluation.clients.common import api_error_metadata, parse_chat_response
from evaluation.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_QWEN_PRESENCE_PENALTY,
    DEFAULT_QWEN_REPETITION_PENALTY,
    DEFAULT_QWEN_TEMPERATURE,
    DEFAULT_QWEN_TOP_K,
    DEFAULT_QWEN_TOP_P,
    SLIDE_TILE_TOOL,
)
from evaluation.protocol import build_messages
from puzzle3.board import Board


class QwenAgent:
    """One-request-per-state Qwen3.5 adapter for a local compatible server."""

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
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)

        self.client = client
        self.model = model
        self.thinking = thinking
        # Accepted for the common runner constructor; Qwen controls thinking with
        # enable_thinking rather than a reasoning-effort level.
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.last_response_metadata: dict[str, Any] = {}

    def next_action(self, board: Board) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(board),
            "tools": [SLIDE_TILE_TOOL],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "extra_body": {
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
                "chat_template_kwargs": {"enable_thinking": self.thinking},
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
                provider="Qwen",
                truncated_reasons={"length"},
            )
        except RuntimeError:
            self.last_response_metadata = {
                "status": "invalid_response",
                "error": "response did not contain a completion choice",
            }
            raise
        return result
