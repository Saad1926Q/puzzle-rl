from __future__ import annotations

from evaluation.protocol import (
    HistoryTurn,
    build_chat_completion_messages,
    build_openai_responses_input,
    build_openrouter_chat_completion_messages,
    parse_tile,
)
from puzzle3.board import GOAL


def test_parser_requires_one_valid_tile() -> None:
    assert parse_tile('{"tile": 8}') == 8
    assert parse_tile('{"tile": 0}') is None
    assert parse_tile('{"tile": 9}') is None
    assert parse_tile('{"tile": true}') is None
    assert parse_tile('{"move": "left"}') is None

def test_messages_contain_current_board_but_no_history() -> None:
    messages = build_chat_completion_messages((1, 2, 3, 4, 5, 6, 7, 0, 8))
    assert len(messages) == 2
    assert "1 2 3" in messages[1]["content"]
    assert "7 0 8" in messages[1]["content"]
    assert "_" not in messages[1]["content"]
    assert "1 2 3 / 4 5 6 / 7 8 0" in messages[0]["content"]
    assert "history" not in messages[1]["content"].lower()

def test_messages_preserve_tool_protocol_and_optional_reasoning() -> None:
    history = (
        HistoryTurn((1, 2, 3, 4, 5, 6, 0, 7, 8), tile=7, reasoning="Move right."),
        HistoryTurn((1, 2, 3, 4, 5, 6, 7, 0, 8), tile=8, reasoning="Solve it."),
    )

    without_reasoning = build_chat_completion_messages(GOAL, history)
    with_reasoning = build_chat_completion_messages(GOAL, history, include_reasoning=True)

    assert [message["role"] for message in without_reasoning] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert "0 7 8" in without_reasoning[1]["content"]
    assert without_reasoning[2]["content"] == ""
    assert without_reasoning[2]["tool_calls"] == [
        {
            "id": "history_slide_0",
            "type": "function",
            "function": {"name": "slide_tile", "arguments": '{"tile": 7}'},
        }
    ]
    assert without_reasoning[3]["tool_call_id"] == "history_slide_0"
    assert "7 0 8" in without_reasoning[3]["content"]
    assert without_reasoning[4]["tool_calls"][0]["function"]["arguments"] == '{"tile": 8}'
    assert without_reasoning[5]["tool_call_id"] == "history_slide_1"
    assert "7 8 0" in without_reasoning[5]["content"]
    assert all(
        "Action: slide tile" not in str(message) for message in without_reasoning
    )
    assert with_reasoning[2]["content"] == "Move right."

def test_openrouter_messages_preserve_native_reasoning_fields() -> None:
    reasoning_details = [
        {"type": "reasoning.text", "text": "Native reasoning", "signature": "sig"}
    ]
    history = (
        HistoryTurn(
            (1, 2, 3, 4, 5, 6, 0, 7, 8),
            tile=7,
            reasoning="Plaintext fallback.",
            reasoning_details=reasoning_details,
        ),
        HistoryTurn(
            (1, 2, 3, 4, 5, 6, 7, 0, 8),
            tile=8,
            reasoning="Solve it.",
        ),
    )

    messages = build_openrouter_chat_completion_messages(
        GOAL, history, include_reasoning=True
    )

    assert messages[2]["content"] == ""
    assert messages[2]["reasoning_details"] is reasoning_details
    assert "reasoning" not in messages[2]
    assert messages[4]["content"] == ""
    assert messages[4]["reasoning"] == "Solve it."
    assert "reasoning_details" not in messages[4]

def test_openrouter_messages_omit_reasoning_when_not_requested() -> None:
    history = (
        HistoryTurn(
            (1, 2, 3, 4, 5, 6, 7, 0, 8),
            tile=8,
            reasoning="Solve it.",
            reasoning_details=[{"type": "reasoning.text", "text": "Solve it."}],
        ),
    )

    message = build_openrouter_chat_completion_messages(GOAL, history)[2]

    assert message["content"] == ""
    assert "reasoning" not in message
    assert "reasoning_details" not in message

def test_responses_input_preserves_function_call_history() -> None:
    history = (HistoryTurn((1, 2, 3, 4, 5, 6, 7, 0, 8), tile=8),)

    items = build_openai_responses_input(GOAL, history)

    assert items[2] == {
        "type": "function_call",
        "call_id": "history_slide_0",
        "name": "slide_tile",
        "arguments": '{"tile": 8}',
    }
    assert items[3]["type"] == "function_call_output"
    assert items[3]["call_id"] == "history_slide_0"
    assert "7 8 0" in items[3]["output"]

def test_responses_input_keeps_reasoning_outside_function_call() -> None:
    history = (
        HistoryTurn(
            (1, 2, 3, 4, 5, 6, 7, 0, 8),
            tile=8,
            reasoning="Move tile 8.",
        ),
    )

    items = build_openai_responses_input(GOAL, history, include_reasoning=True)

    assert items[2] == {"role": "assistant", "content": "Move tile 8."}
    assert items[3]["type"] == "function_call"
    assert "content" not in items[3]

