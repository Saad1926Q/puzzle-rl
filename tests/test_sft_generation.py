from __future__ import annotations

import json


import pytest

from evaluation.dataset import PuzzleExample, load_examples
from evaluation.protocol import build_chat_completion_messages
from puzzle3.board import GOAL
from sft_generation.annotation import (
    annotate_step,
    annotation_messages,
    clean_rationale,
    validate_annotation,
)
from sft_generation.client import TextCompletion
from sft_generation.formatting import build_sft_record
from sft_generation.rollout import (
    RolloutConfig,
    run_rollout,
    trajectory_record,
    validate_trajectory,
)


BOARD = (1, 2, 3, 4, 5, 6, 0, 7, 8)
EXAMPLE = PuzzleExample("source-1", BOARD, (7, 8), 2, {})

def test_source_dataset_does_not_require_optimal_actions(tmp_path) -> None:
    path = tmp_path / "source.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "source-1",
                "board": list(BOARD),
                "action_interface": "tile_id_v1",
                "optimal_length": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_examples(
        dataset=str(path),
        require_optimal_actions=False,
    )

    assert examples[0].optimal_actions == ()
    assert examples[0].optimal_length == 2


def trajectory() -> dict:
    return {
        "source_id": "source-1",
        "initial_board": list(BOARD),
        "optimal_length": 2,
        "rollout_id": 1,
        "moves_taken": 2,
        "final_board": list(GOAL),
        "steps": [
            {
                "turn": 1,
                "board": list(BOARD),
                "legal_tiles": [4, 7],
                "tile": 7,
                "next_board": [1, 2, 3, 4, 5, 6, 7, 0, 8],
            },
            {
                "turn": 2,
                "board": [1, 2, 3, 4, 5, 6, 7, 0, 8],
                "legal_tiles": [5, 7, 8],
                "tile": 8,
                "next_board": list(GOAL),
            },
        ],
    }


def test_validate_trajectory_replays_every_move() -> None:
    validate_trajectory(trajectory())
    invalid = trajectory()
    invalid["steps"][0]["tile"] = 5
    with pytest.raises(ValueError, match="illegal tile"):
        validate_trajectory(invalid)


def test_custom_system_prompt_replaces_default_prompt() -> None:
    messages = build_chat_completion_messages(BOARD, system_prompt="sft prompt")
    assert messages[0] == {"role": "system", "content": "sft prompt"}


def test_run_rollout_passes_sft_prompt_and_keeps_reasoning() -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self.actions = iter([json.dumps({"tile": 7}), json.dumps({"tile": 8})])
            self.last_response_metadata = {"reasoning_content": "move"}

        def next_action(self, board, history=(), **kwargs):
            return next(self.actions)

    created = {}

    def factory(**kwargs):
        created.update(kwargs)
        return FakeAgent()

    episode = run_rollout(
        EXAMPLE,
        rollout_id=2,
        api_key="unused",
        config=RolloutConfig(max_tokens=100),
        client_factory=factory,
    )
    assert episode.solved
    assert episode.rollout_id == 2
    assert "Solve systematically in stages" in created["system_prompt"]


def test_trajectory_record_contains_verified_steps() -> None:
    class Agent:
        def __init__(self) -> None:
            self.actions = iter([json.dumps({"tile": 7}), json.dumps({"tile": 8})])
            self.last_response_metadata = {"reasoning_content": "move"}

        def next_action(self, board, history=(), **kwargs):
            return next(self.actions)

    episode = run_rollout(
        EXAMPLE,
        rollout_id=1,
        api_key="unused",
        config=RolloutConfig(),
        client_factory=lambda **kwargs: Agent(),
    )
    record = trajectory_record(episode)
    assert record["steps"][1]["next_board"] == list(GOAL)
    assert record["steps"][0]["reasoning"] == "move"


def test_annotation_prompt_contains_verified_move_context() -> None:
    messages = annotation_messages(trajectory(), 0)
    assert "Verified selected tile: 7" in messages[1]["content"]
    assert "exact distance" in messages[0]["content"]


def test_annotation_is_validated_without_changing_action() -> None:
    class Client:
        def complete(self, messages):
            return TextCompletion("Sliding tile 7 places it into the open bottom-middle goal position.", {})

    result = annotate_step(trajectory(), 0, client=Client())
    assert result["valid"] is True
    assert result["tile"] == 7
    assert clean_rationale("<think>  A useful move. </think>") == "A useful move."
    bad = {
        **result,
        "rationale": "The solver selected this move because it is optimal for solving.",
    }
    with pytest.raises(ValueError, match="forbidden"):
        validate_annotation(bad, trajectory(), 0)
    long = {**result, "rationale": " ".join(["Move"] * 31)}
    with pytest.raises(ValueError, match="10 to 30 words"):
        validate_annotation(long, trajectory(), 0)


def test_sft_record_preserves_verified_tool_call_and_roles() -> None:
    annotations = [
        {
            "source_id": "source-1",
            "turn": 1,
            "board": list(BOARD),
            "tile": 7,
            "next_board": [1, 2, 3, 4, 5, 6, 7, 0, 8],
            "rationale": "Sliding tile 7 places it into the open bottom-middle goal position.",
            "valid": True,
        },
        {
            "source_id": "source-1",
            "turn": 2,
            "board": [1, 2, 3, 4, 5, 6, 7, 0, 8],
            "tile": 8,
            "next_board": list(GOAL),
            "rationale": "Sliding tile 8 completes the final lower-right placement and solves the puzzle.",
            "valid": True,
        },
    ]
    record = build_sft_record(trajectory(), annotations)
    assert record["prompt"][0]["role"] == "system"
    assert [item["role"] for item in record["completion"]] == [
        "assistant", "tool", "assistant", "tool"
    ]
    call = record["completion"][0]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"tile": 7}
    assert "<think>" in record["completion"][0]["content"]
