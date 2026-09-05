"""Provider-neutral 8-puzzle model protocol and response parsing."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from dotenv import load_dotenv

from evaluation.constants import (
    DEFAULT_API_KEY_ENV,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_WITH_HISTORY,
)
from puzzle3.board import Board
from puzzle3.render import render


@dataclass(frozen=True)
class HistoryTurn:
    """One completed environment transition available as future-turn context."""

    board: Board
    tile: int
    reasoning: str = ""
    reasoning_details: Any | None = None


class PuzzleAgent(Protocol):
    """Minimal interface required by the stateful evaluator."""

    def next_action(
        self,
        board: Board,
        history: Sequence[HistoryTurn] = (),
        *,
        include_reasoning: bool = False,
    ) -> str:
        """Return one canonical tile-selection response for the current board."""


def _board_prompt(board: Board, *, after_action: bool = False) -> str:
    heading = (
        "Board after that action (0 is the blank):"
        if after_action
        else "Current board (0 is the blank):"
    )
    return "\n".join(
        (
            heading,
            render(board),
            "\nChoose the single adjacent numbered tile to slide into the blank now.",
        )
    )


def build_chat_completion_messages(
    board: Board,
    history: Sequence[HistoryTurn] = (),
    *,
    include_reasoning: bool = False,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build Chat Completions messages for one puzzle state."""

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt
            or (SYSTEM_PROMPT_WITH_HISTORY if history else SYSTEM_PROMPT),
        }
    ]
    if not history:
        messages.append({"role": "user", "content": _board_prompt(board)})
        return messages

    messages.append({"role": "user", "content": _board_prompt(history[0].board)})
    for index, turn in enumerate(history):
        call_id = f"history_slide_{index}"
        messages.append(
            {
                "role": "assistant",
                "content": turn.reasoning if include_reasoning and turn.reasoning else "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "slide_tile",
                            "arguments": json.dumps({"tile": turn.tile}),
                        },
                    }
                ],
            }
        )
        next_board = history[index + 1].board if index + 1 < len(history) else board
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": _board_prompt(next_board, after_action=True),
            }
        )
    return messages


def build_openrouter_chat_completion_messages(
    board: Board,
    history: Sequence[HistoryTurn] = (),
    *,
    include_reasoning: bool = False,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build OpenRouter messages while preserving native reasoning fields."""

    messages = build_chat_completion_messages(
        board,
        history,
        include_reasoning=include_reasoning,
        system_prompt=system_prompt,
    )
    if not include_reasoning:
        return messages

    assistant_messages = (
        message for message in messages if message["role"] == "assistant"
    )
    for message, turn in zip(assistant_messages, history, strict=True):
        message["content"] = ""
        if turn.reasoning_details:
            message["reasoning_details"] = turn.reasoning_details
        elif turn.reasoning:
            message["reasoning"] = turn.reasoning
    return messages


def build_openai_responses_input(
    board: Board,
    history: Sequence[HistoryTurn] = (),
    *,
    include_reasoning: bool = False,
) -> list[dict[str, Any]]:
    """Build the OpenAI Responses API wire format for the same puzzle context.

    Unlike Chat Completions, this API represents retained actions as standalone
    ``function_call`` and ``function_call_output`` items. Both builders accept
    the same logical board/history inputs but serialize them for different APIs.
    """

    if not history:
        return build_chat_completion_messages(board)

    items: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT_WITH_HISTORY},
        {"role": "user", "content": _board_prompt(history[0].board)},
    ]
    for index, turn in enumerate(history):
        call_id = f"history_slide_{index}"
        if include_reasoning and turn.reasoning:
            items.append({"role": "assistant", "content": turn.reasoning})
        items.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": "slide_tile",
                "arguments": json.dumps({"tile": turn.tile}),
            }
        )
        next_board = history[index + 1].board if index + 1 < len(history) else board
        items.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": _board_prompt(next_board, after_action=True),
            }
        )
    return items


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
