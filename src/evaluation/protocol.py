"""Provider-neutral 8-puzzle model protocol and response parsing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

from evaluation.constants import DEFAULT_API_KEY_ENV, SYSTEM_PROMPT
from puzzle3.render import render


class MoveAgent(Protocol):
    """Minimal interface required by the stateful evaluator."""

    def next_move(self, board: tuple[int, ...]) -> str:
        """Return one canonical tile-selection response for the current board."""


def build_messages(board: tuple[int, ...]) -> list[dict[str, str]]:
    """Build a fresh request; no previous state or action is included."""

    board_text = "\n".join(
        (
            "Current board (0 is the blank):",
            render(board),
            "\nChoose the single adjacent numbered tile to slide into the blank now.",
        )
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": board_text},
    ]


def _valid_tile(value: Any) -> int | None:
    """Return a valid numbered tile ID, explicitly excluding booleans."""

    return value if type(value) is int and 1 <= value <= 8 else None


def parse_tile(response: str | None) -> int | None:
    """Extract one numbered tile ID from canonical JSON text."""

    if not isinstance(response, str):
        return None
    try:
        payload = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return None
    return _valid_tile(payload.get("tile") if isinstance(payload, dict) else None)


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


def _parse_tool_tile(name: Any, arguments: Any) -> int | None:
    if name != "slide_tile" or not isinstance(arguments, str):
        return None
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    return _valid_tile(payload.get("tile") if isinstance(payload, dict) else None)


def extract_chat_tool_tile(message: Any) -> tuple[int | None, dict[str, Any]]:
    """Extract exactly one tile call from a Chat Completions message."""

    calls = getattr(message, "tool_calls", None) or []
    metadata: dict[str, Any] = {"tool_call_count": len(calls)}
    if len(calls) != 1:
        metadata["invalid_tool_call_count"] = True
        return None, metadata
    function = getattr(calls[0], "function", None)
    name = getattr(function, "name", None)
    arguments = getattr(function, "arguments", None)
    metadata.update({"tool_name": name, "tool_arguments": arguments})
    return _parse_tool_tile(name, arguments), metadata


def extract_responses_tool_tile(response: Any) -> tuple[int | None, dict[str, Any]]:
    """Extract exactly one tile call from a Responses API response."""

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
    return _parse_tool_tile(name, arguments), metadata
