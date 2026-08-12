"""DeepSeek-compatible model adapter and robust move extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

from evaluation.constants import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_THINKING,
    MOVE_TAG_RE,
    MOVE_TOOL,
    SYSTEM_PROMPT,
    VALID_MOVE_SET,
)
from puzzle3.render import render


class MoveAgent(Protocol):
    """Minimal interface required by the stateful evaluator."""

    def next_move(self, board: tuple[int, ...]) -> str:
        """Return one canonical move response for the current board."""


def build_messages(board: tuple[int, ...]) -> list[dict[str, str]]:
    """Build a fresh request; no previous state or action is included."""

    board_text = "\n".join(
        (
            "Current board (0 is the blank):",
            render(board),
            "\nChoose the single next move now.",
        )
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": board_text},
    ]


def parse_move(response: str | None) -> str | None:
    """Extract a move from legacy XML or JSON text, or return ``None``."""

    if not isinstance(response, str):
        return None
    matches = MOVE_TAG_RE.findall(response)
    if len(matches) == 1:
        return matches[0].lower()
    try:
        payload = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(payload, dict) and isinstance(payload.get("move"), str):
        move = payload["move"].strip().lower()
        return move if move in VALID_MOVE_SET else None
    return None


def get_api_key(
    env_name: str = DEFAULT_API_KEY_ENV, dotenv_path: str | Path = ".env"
) -> str:
    load_dotenv(dotenv_path=dotenv_path, override=False)
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(
            f"Missing {env_name}; set it in {dotenv_path} or export it in the environment"
        )
    return key


def _value(value: Any) -> Any:
    """Convert SDK model values to JSON-safe values for diagnostics."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _value(value.model_dump())
    if hasattr(value, "__dict__"):
        return _value(vars(value))
    return str(value)


def _tool_call_move(message: Any) -> tuple[str | None, dict[str, Any]]:
    """Extract and validate the strict submit_move call from an SDK message."""

    calls = getattr(message, "tool_calls", None) or []
    metadata: dict[str, Any] = {"tool_call_count": len(calls)}
    if len(calls) != 1:
        metadata["invalid_tool_call_count"] = True
        return None, metadata
    call = calls[0]
    function = getattr(call, "function", None)
    name = getattr(function, "name", None)
    arguments = getattr(function, "arguments", None)
    metadata.update({"tool_name": name, "tool_arguments": arguments})
    if name != "submit_move" or not isinstance(arguments, str):
        return None, metadata
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError:
        return None, metadata
    move = payload.get("move") if isinstance(payload, dict) else None
    if not isinstance(move, str) or move.lower() not in VALID_MOVE_SET:
        return None, metadata
    return move.lower(), metadata


class DeepSeekAgent:
    """One-request-per-state adapter using DeepSeek's strict function interface.

    DeepSeek thinking mode is enabled by default by the API, but a forced named tool
    call is not supported in thinking mode. The default therefore disables thinking
    and forces ``submit_move``: this gives the evaluator a schema-validated action
    rather than fragile XML. ``thinking=enabled`` is supported with ``tool_choice``
    set to ``auto`` and still consumes a returned tool call when one is produced.
    """

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
        if reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("reasoning_effort must be low, high, or max")
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
            "tool_choice": "auto"
            if self.thinking
            else {
                "type": "function",
                "function": {"name": "submit_move"},
            },
            "max_tokens": self.max_tokens,
        }
        if self.thinking:
            request["reasoning_effort"] = self.reasoning_effort
            request["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            request["extra_body"] = {"thinking": {"type": "disabled"}}

        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            self.last_response_metadata = {
                "status": "api_error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
            raise

        try:
            choice = response.choices[0]
            message = choice.message
        except (AttributeError, IndexError, TypeError) as exc:
            self.last_response_metadata = {
                "status": "invalid_response",
                "error": "response did not contain a completion choice",
            }
            raise RuntimeError(
                "DeepSeek response did not contain a completion choice"
            ) from exc

        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        if not isinstance(content, str):
            content = ""
        reasoning = getattr(message, "reasoning_content", None)
        if not isinstance(reasoning, str):
            reasoning = ""
        tool_move, tool_metadata = _tool_call_move(message)
        finish_reason = getattr(choice, "finish_reason", None)
        self.last_response_metadata = {
            "status": "tool_call" if tool_move else "malformed",
            "finish_reason": finish_reason,
            "content": content,
            "content_length": len(content),
            "reasoning_content_length": len(reasoning),
            # This is the complete reasoning_content returned by the API. It is
            # intentionally persisted so the output JSON is a full trajectory trace.
            "reasoning_content": reasoning,
            "reasoning_preview": reasoning[:1000],
            "truncated": finish_reason == "length",
            **tool_metadata,
            "usage": _value(getattr(response, "usage", None)),
        }
        if finish_reason == "length":
            # A tool payload accompanied by a length stop is not a reliable complete
            # completion; do not silently accept a possibly truncated action.
            self.last_response_metadata["status"] = "truncated"
            return ""
        if tool_move is not None:
            # Keep the evaluator interface simple while retaining the real tool data.
            return json.dumps({"move": tool_move})
        # Fallback supports compatible endpoints that return text instead of a tool call.
        return content
