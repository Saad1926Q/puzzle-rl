from __future__ import annotations

import json

from puzzle3.environment import PuzzleEnv
from rl_training.sync_rollout import EpisodeRollout, TurnRecord
from rl_training.trajectory import RLTrajectoryWriter


def test_writer_appends_one_record_per_group(tmp_path) -> None:
    environment = PuzzleEnv()
    environment.reset(
        board=[1, 2, 3, 4, 5, 6, 0, 7, 8],
        optimal_length=2,
        max_turns=2,
    )
    episode = EpisodeRollout(
        environment=environment,
        turns=[
            TurnRecord(
                prompt_ids=[1],
                completion_ids=[2],
                logprobs=[-0.1],
                board=(1, 2, 3, 4, 5, 6, 0, 7, 8),
                legal_tiles=(7,),
                reasoning="Move the tile next to the blank.",
                move=7,
                next_board=(1, 2, 3, 4, 5, 6, 7, 0, 8),
                status="valid",
                reward=0.0,
            )
        ],
    )
    path = tmp_path / "trajectories.jsonl"

    with RLTrajectoryWriter(path) as writer:
        writer.write_group(
            training_step=4,
            group=3,
            episodes=[episode],
            advantages=[0.5],
        )

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {
            "attempts": [
                {
                    "advantage": 0.5,
                    "initial_board": [1, 2, 3, 4, 5, 6, 0, 7, 8],
                    "max_turns": 2,
                    "optimal_length": 2,
                    "outcome": "running",
                    "reward": 0.0,
                    "tool_calls": 0,
                    "tool_failures": 0,
                    "truncated": False,
                    "turns": [
                        {
                            "board": [1, 2, 3, 4, 5, 6, 0, 7, 8],
                            "legal_tiles": [7],
                            "move": 7,
                            "next_board": [1, 2, 3, 4, 5, 6, 7, 0, 8],
                            "progress_reward": 0.0,
                            "reasoning": "Move the tile next to the blank.",
                            "reasoning_details": None,
                            "reward": 0.0,
                            "status": "valid",
                            "terminal_reward": 0.0,
                            "turn": 1,
                        }
                    ],
                }
            ],
            "group": 3,
            "training_step": 4,
        }
    ]


def test_writer_rejects_existing_path(tmp_path) -> None:
    path = tmp_path / "trajectories.jsonl"
    path.write_text("existing\n")

    try:
        RLTrajectoryWriter(path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("writer should not overwrite an existing trajectory file")
