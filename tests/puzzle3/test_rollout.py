from __future__ import annotations

from evaluation.protocol import build_chat_completion_messages
from puzzle3.history import LastTurns
from rl_training.rollout import PuzzleRolloutEngine


BOARD = (1, 2, 3, 4, 5, 6, 7, 0, 8)


def _assistant(tile: int) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": f"Move {tile}.",
        "tool_calls": [
            {
                "id": f"call-{tile}",
                "type": "function",
                "function": {"name": "slide_tile", "arguments": {"tile": tile}},
            }
        ],
    }


def test_rollout_engine_uses_initial_prompt_then_bounded_history() -> None:
    initial = build_chat_completion_messages(BOARD)
    engine = PuzzleRolloutEngine(initial, history_policy=LastTurns(4))

    assert engine.messages(BOARD) == initial

    for turn in range(1, 6):
        engine.record_action(board=BOARD, tile=turn, assistant_message=_assistant(turn))

    messages = engine.messages(BOARD)
    assistant_messages = [message for message in messages if message["role"] == "assistant"]
    user_message = next(message for message in messages if message["role"] == "user")

    assert len(assistant_messages) == 4
    assert [message["tool_calls"][0]["function"]["arguments"] for message in assistant_messages] == [
        '{"tile": 2}',
        '{"tile": 3}',
        '{"tile": 4}',
        '{"tile": 5}',
    ]
    assert "Current board" in user_message["content"]


def test_rollout_engine_returns_fresh_messages() -> None:
    initial = build_chat_completion_messages(BOARD)
    engine = PuzzleRolloutEngine(initial)

    first = engine.messages(BOARD)
    first[0]["content"] = "mutated"

    assert engine.messages(BOARD)[0]["content"] != "mutated"
