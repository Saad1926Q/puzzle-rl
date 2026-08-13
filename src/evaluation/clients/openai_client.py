"""OpenAI Responses API client."""

from __future__ import annotations

import json
from typing import Any

from evaluation.clients.common import (
    api_error_metadata,
    empty_reasoning_metadata,
    validate_reasoning_effort,
)
from evaluation.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REASONING_EFFORT,
    MOVE_TOOL,
)
from evaluation.protocol import build_messages, extract_responses_tool_move, json_safe


class OpenAIAgent:
    """One-request-per-state OpenAI Responses API adapter."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        thinking: bool = True,
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
        tool = {
            "type": "function",
            "name": "submit_move",
            "description": MOVE_TOOL["function"]["description"],
            "parameters": MOVE_TOOL["function"]["parameters"],
            "strict": True,
        }
        request: dict[str, Any] = {
            "model": self.model,
            "input": build_messages(board),
            "tools": [tool],
            "tool_choice": {"type": "function", "name": "submit_move"},
            "parallel_tool_calls": False,
            "max_output_tokens": self.max_tokens,
            "store": False,
        }
        if self.thinking:
            request["reasoning"] = {"effort": self.reasoning_effort}
        try:
            response = self.client.responses.create(**request)
        except Exception as exc:
            self.last_response_metadata = api_error_metadata(exc)
            raise
        incomplete = getattr(response, "status", None) == "incomplete"
        move, tool_metadata = extract_responses_tool_move(response)
        self.last_response_metadata = {
            "status": (
                "truncated" if incomplete else ("tool_call" if move else "malformed")
            ),
            "finish_reason": None,
            "incomplete_reason": getattr(
                getattr(response, "incomplete_details", None), "reason", None
            ),
            "content": getattr(response, "output_text", "") or "",
            "content_length": len(getattr(response, "output_text", "") or ""),
            **empty_reasoning_metadata(),
            "truncated": incomplete,
            **tool_metadata,
            "usage": json_safe(getattr(response, "usage", None)),
        }
        if incomplete:
            return ""
        return json.dumps({"move": move}) if move else (getattr(response, "output_text", "") or "")
