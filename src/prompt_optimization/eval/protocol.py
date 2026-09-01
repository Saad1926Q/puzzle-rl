"""Prompt-optimization-specific OpenRouter message and response handling."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence
from dotenv import load_dotenv

from puzzle3.board import Board
from puzzle3.render import render


@dataclass(frozen=True)
class HistoryTurn:
    board: Board
    tile: int
    reasoning: str = ""
    reasoning_details: Any | None = None


class PuzzleAgent(Protocol):
    def next_action(
        self,
        board: Board,
        history: Sequence[HistoryTurn] = (),
        *,
        include_reasoning: bool = False,
    ) -> str: ...


def board_prompt(board: Board, *, after_action: bool = False) -> str:
    heading = "Board after that action (0 is the blank):" if after_action else "Current board (0 is the blank):"
    return f"{heading}\n{render(board)}\n\nChoose the single adjacent numbered tile to slide into the blank now."


def build_messages(
    board: Board,
    system_prompt: str,
    history: Sequence[HistoryTurn] = (),
    *,
    include_reasoning: bool = False,
) -> list[dict[str, Any]]:
    """Build one candidate-specific OpenRouter Chat Completions request."""

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if not history:
        messages.append({"role": "user", "content": board_prompt(board)})
        return messages

    messages.append({"role": "user", "content": board_prompt(history[0].board)})
    for index, turn in enumerate(history):
        call_id = f"history_slide_{index}"
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": turn.reasoning if include_reasoning and turn.reasoning else "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": "slide_tile", "arguments": json.dumps({"tile": turn.tile})},
            }],
        }
        if include_reasoning:
            if turn.reasoning_details:
                assistant["reasoning_details"] = turn.reasoning_details
            elif turn.reasoning:
                assistant["reasoning"] = turn.reasoning
        messages.append(assistant)
        next_board = history[index + 1].board if index + 1 < len(history) else board
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": board_prompt(next_board, after_action=True),
        })
    return messages


def parse_tile(response: str | None) -> int | None:
    if not isinstance(response, str):
        return None
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return None
    value = payload.get("tile") if isinstance(payload, dict) else None
    return value if type(value) is int and 1 <= value <= 8 else None


def json_safe(value: Any) -> Any:
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


def get_api_key(env_name: str, dotenv_path: str | Path = ".env") -> str:
    load_dotenv(dotenv_path=dotenv_path, override=False)
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(f"Missing {env_name}; set it in {dotenv_path} or export it")
    return key
