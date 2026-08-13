"""Provider-neutral 8-puzzle model protocol and response parsing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

from evaluation.constants import (
    DEFAULT_API_KEY_ENV,
    MOVE_TAG_RE,
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
    """Extract a move from legacy XML or canonical JSON text."""

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
    """Read an API key, giving exported environment variables precedence."""

    load_dotenv(dotenv_path=dotenv_path, override=False)
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(
            f"Missing {env_name}; set it in {dotenv_path} or export it in the environment"
        )
    return key


def json_safe(value: Any) -> Any:
    """Convert SDK model values to JSON-safe diagnostic data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return str(value)


def extract_chat_tool_move(message: Any) -> tuple[str | None, dict[str, Any]]:
    """Extract exactly one function call from a Chat Completions message."""

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


def extract_responses_tool_move(response: Any) -> tuple[str | None, dict[str, Any]]:
    """Extract exactly one function call from a Responses API response."""

    calls = [
        item
        for item in (getattr(response, "output", None) or [])
        if getattr(item, "type", None) == "function_call"
    ]
    metadata: dict[str, Any] = {"tool_call_count": len(calls)}
    if len(calls) != 1:
        metadata["invalid_tool_call_count"] = True
        return None, metadata
    call = calls[0]
    name = getattr(call, "name", None)
    arguments = getattr(call, "arguments", None)
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
