"""DeepSeek Chat Completions client."""

from __future__ import annotations

import json
from typing import Any

from evaluation.clients.common import (
    api_error_metadata,
    empty_reasoning_metadata,
    validate_reasoning_effort,
)
from evaluation.constants import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_THINKING,
    MOVE_TOOL,
)
from evaluation.protocol import build_messages, extract_chat_tool_move, json_safe


class DeepSeekAgent:
    """One-request-per-state DeepSeek adapter."""

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
        self.client = client
        self.model = model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.last_response_metadata: dict[str, Any] = {}

    def next_move(self, board: tuple[int, ...]) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(board),
            "tools": [MOVE_TOOL],
            "tool_choice": (
                "auto"
                if self.thinking
                else {"type": "function", "function": {"name": "submit_move"}}
            ),
            "max_tokens": self.max_tokens,
            "extra_body": {"thinking": {"type": "enabled" if self.thinking else "disabled"}},
        }
        if self.thinking:
            request["reasoning_effort"] = self.reasoning_effort
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            self.last_response_metadata = api_error_metadata(exc)
            raise
        try:
            choice = response.choices[0]
            message = choice.message
        except (AttributeError, IndexError, TypeError) as exc:
            self.last_response_metadata = {
                "status": "invalid_response",
                "error": "response did not contain a completion choice",
            }
            raise RuntimeError("DeepSeek response did not contain a completion choice") from exc
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        if not isinstance(content, str):
            content = ""
        reasoning = getattr(message, "reasoning_content", "")
        if not isinstance(reasoning, str):
            reasoning = ""
        move, tool_metadata = extract_chat_tool_move(message)
        finish_reason = getattr(choice, "finish_reason", None)
        self.last_response_metadata = {
            "status": "tool_call" if move else "malformed",
            "finish_reason": finish_reason,
            "content": content,
            "content_length": len(content),
            **empty_reasoning_metadata(reasoning),
            "truncated": finish_reason == "length",
            **tool_metadata,
            "usage": json_safe(getattr(response, "usage", None)),
        }
        if finish_reason == "length":
            self.last_response_metadata["status"] = "truncated"
            return ""
        return json.dumps({"move": move}) if move else content
