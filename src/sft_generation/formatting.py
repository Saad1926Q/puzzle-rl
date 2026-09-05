"""Build SFT conversations from verified actions and rationales."""

from __future__ import annotations

import json
from typing import Any, Iterable

from puzzle3.render import render
from sft_generation.annotation import validate_annotation
from sft_generation.constants import SFT_SOLVING_PROMPT_WITH_HISTORY
from sft_generation.rollout import validate_trajectory


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
    trajectory: dict[str, Any],
    annotations: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Build one prompt/completion conversation for a solved puzzle."""

    validate_trajectory(trajectory)
    annotation_map = {(item.get("source_id"), item.get("turn")): item for item in annotations}
    completion: list[dict[str, Any]] = []
    for step_index, step in enumerate(trajectory["steps"]):
        key = (trajectory["source_id"], step["turn"])
        annotation = annotation_map.get(key)
        if annotation is None or not annotation.get("valid"):
            raise ValueError(f"missing valid annotation for {key}")
        validate_annotation(annotation, trajectory, step_index)
        call_id = f"sft_{trajectory['source_id']}_{step['turn']}"
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
                            "arguments": json.dumps({"tile": step["tile"]}),
                        },
                    }
                ],
            }
        )
        completion.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": board_message(step["next_board"], after_action=True),
            }
        )
    return {
        "prompt": [
            {"role": "system", "content": SFT_SOLVING_PROMPT_WITH_HISTORY},
            {"role": "user", "content": board_message(trajectory["initial_board"])},
        ],
        "completion": completion,
        "metadata": {
            "source_id": trajectory["source_id"],
            "initial_board": trajectory["initial_board"],
            "optimal_length": trajectory["optimal_length"],
            "moves_taken": trajectory["moves_taken"],
            "rollout_id": trajectory["rollout_id"],
            "teacher_model": trajectory.get("teacher_model"),
        },
    }
