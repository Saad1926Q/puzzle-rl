from __future__ import annotations

import pytest

from puzzle3.board import GOAL
from rl_training.environment import PuzzleEnvironment


ONE_MOVE = (1, 2, 3, 4, 5, 6, 7, 0, 8)


def test_environment_reset_accepts_dataset_columns_and_exposes_observation() -> None:
    environment = PuzzleEnvironment()

    observation = environment.reset(
        board=list(ONE_MOVE),
        optimal_length=1,
        prompt=[{"role": "user", "content": "Solve it."}],
    )

    assert "1 2 3" in observation
    assert "Legal tiles: [5, 7, 8]" in observation
    assert environment.done is False
    assert environment.outcome == "running"


def test_environment_tool_solves_and_reports_reward() -> None:
    environment = PuzzleEnvironment()
    environment.reset(board=ONE_MOVE, optimal_length=1)

    observation = environment.slide_tile(8)

    assert "Status: solved" in observation
    assert environment.done is True
    assert environment.episode is not None
    assert environment.episode.board == GOAL
    assert environment.get_reward() == 1.0


def test_environment_illegal_tool_call_terminates_episode() -> None:
    environment = PuzzleEnvironment()
    environment.reset(board=ONE_MOVE, optimal_length=1)

    with pytest.raises(ValueError, match="illegal tile 1"):
        environment.slide_tile(1)

    assert environment.done is True
    assert environment.outcome == "illegal"
    assert environment.get_reward() == -1.0


def test_environment_requires_reset_before_tool_use() -> None:
    with pytest.raises(RuntimeError, match="reset"):
        PuzzleEnvironment().slide_tile(8)
