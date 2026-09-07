"""Build SFT conversations from verified actions and rationales."""

import json
from typing import Any, Iterable

from evaluation.constants import SLIDE_TILE_TOOL, SYSTEM_PROMPT_WITH_HISTORY
from evaluation.protocol import HistoryTurn, build_chat_completion_messages
from puzzle3.render import render
from sft_generation.annotation import validate_annotation
from sft_generation.records import Trajectory


def board_message(board: list[int] | tuple[int, ...], *, after_action: bool = False) -> str:
    """Render one board as a task or tool-result message."""
    heading = "Board after that action" if after_action else "Current board"
    return "\n".join(
        (
            f"{heading} (0 is the blank):",
            render(tuple(board)),
            "Choose the single adjacent numbered tile to slide into the blank now.",
        )
    )



def build_sft_record(
    trajectory: Trajectory,
    annotations: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Build one prompt/completion conversation for a solved puzzle."""

    annotation_map = {(item.get("source_id"), item.get("turn")): item for item in annotations}
    completion: list[dict[str, Any]] = []
    for step_index, step in enumerate(trajectory.steps):
        key = (trajectory.source_id, step.turn)
        annotation = annotation_map.get(key)
        if annotation is None or not annotation.get("valid"):
            raise ValueError(f"missing valid annotation for {key}")
        validate_annotation(annotation, trajectory, step_index)
        call_id = f"sft_{trajectory.source_id}_{step.turn}"
        completion.append(
            {
                "role": "assistant",
                "content": f"<think>\n{annotation['rationale']}\n</think>",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "slide_tile",
                            "arguments": json.dumps({"tile": step.tile}),
                        },
                    }
                ],
            }
        )
        completion.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": board_message(step.next_board, after_action=True),
            }
        )
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT_WITH_HISTORY},
            {"role": "user", "content": board_message(trajectory.initial_board)},
        ],
        "completion": completion,
        "metadata": {
            "source_id": trajectory.source_id,
            "initial_board": list(trajectory.initial_board),
            "optimal_length": trajectory.optimal_length,
            "moves_taken": trajectory.moves_taken,
            "rollout_id": trajectory.rollout_id,
            "teacher_model": trajectory.teacher_model,
        },
    }

def build_history_window_sft_records(
    trajectory: Trajectory,
    annotations: Iterable[dict[str, Any]],
    *,
    history_turns: int = 4,
) -> list[dict[str, Any]]:
    """Build one SFT record per move with bounded evaluator history."""
    if history_turns < 0:
        raise ValueError("history_turns must be non-negative")

    annotation_map = {
        (item.get("source_id"), item.get("turn")): item for item in annotations
    }
    steps = trajectory.steps
    records: list[dict[str, Any]] = []

    for step_index, step in enumerate(steps):
        target_key = (trajectory.source_id, step.turn)
        target_annotation = annotation_map.get(target_key)
        if target_annotation is None or not target_annotation.get("valid"):
            raise ValueError(f"missing valid annotation for {target_key}")
        validate_annotation(target_annotation, trajectory, step_index)

        retained_steps = steps[max(0, step_index - history_turns) : step_index]
        history = []
        for retained_step in retained_steps:
            retained_key = (
                trajectory.source_id,
                retained_step.turn,
            )
            retained_annotation = annotation_map.get(retained_key)
            if retained_annotation is None or not retained_annotation.get("valid"):
                raise ValueError(f"missing valid annotation for {retained_key}")
            history.append(
                HistoryTurn(
                    board=retained_step.board,
                    tile=retained_step.tile,
                    reasoning=(
                        f"<think>\n{retained_annotation['rationale']}\n</think>"
                    ),
                )
            )

        prompt = build_chat_completion_messages(
            step.board,
            history,
            include_reasoning=True,
        )
        call_id = f"sft_{trajectory.source_id}_{step.turn}"
        records.append(
            {
                "prompt": prompt,
                "completion": [
                    {
                        "role": "assistant",
                        "content": (
                            f"<think>\n{target_annotation['rationale']}\n</think>"
                        ),
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "slide_tile",
                                    "arguments": json.dumps(
                                        {"tile": step.tile}
                                    ),
                                },
                            }
                        ],
                    }
                ],
                "metadata": {
                    "record_type": "decision",
                    "source_id": trajectory.source_id,
                    "rollout_id": trajectory.rollout_id,
                    "target_turn": step.turn,
                    "history_turns": len(retained_steps),
                    "initial_board": list(trajectory.initial_board),
                    "initial_depth": trajectory.optimal_length,
                    "board": list(step.board),
                    "legal_tiles": list(step.legal_tiles),
                    "tile": step.tile,
                    "next_board": list(step.next_board),
                    "teacher_model": trajectory.teacher_model,
                },
                "tools": [SLIDE_TILE_TOOL],
            }
        )
    return records
