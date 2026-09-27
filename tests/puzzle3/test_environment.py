from __future__ import annotations

import pytest

from puzzle3.board import GOAL
from puzzle3.environment import PuzzleEnv
from puzzle3.rewards import distance_progress_reward, solved_reward

ONE_MOVE = (1, 2, 3, 4, 5, 6, 7, 0, 8)
TWO_MOVES = (1, 2, 3, 4, 5, 6, 0, 7, 8)


def make_env(
    board: tuple[int, ...], optimal_length: int, max_turns: int = 45
) -> PuzzleEnv:
    env = PuzzleEnv()
    env.reset(board, optimal_length, max_turns)
    return env


def test_legal_move_solves_with_terminal_reward() -> None:
    env = make_env(ONE_MOVE, 1)

    result = env._move(8)

    assert result.status == "solved"
    assert env.board == GOAL
    assert env.done is True
    assert env.reward == solved_reward(1, 1)
    assert env.progress_reward == distance_progress_reward(ONE_MOVE, GOAL)


def test_timeout_keeps_progress_reward() -> None:
    env = make_env(TWO_MOVES, 2, max_turns=1)

    result = env._move(7)

    assert result.status == "timeout"
    assert env.reward == result.progress_reward
    assert env.done is True


def test_illegal_move_preserves_progress_and_applies_penalty() -> None:
    env = make_env(TWO_MOVES, 2)
    first = env._move(7)

    result = env._move(1)

    assert result.status == "illegal"
    assert env.outcome == "illegal"
    assert env.reward == env.progress_reward - 1.0
    assert env.reward == first.progress_reward - 1.0
    assert env.progress_reward > 0.0


def test_malformed_response_preserves_progress_and_applies_penalty() -> None:
    env = make_env(TWO_MOVES, 2)
    first = env._move(7)

    result = env._fail("malformed")

    assert result.status == "malformed"
    assert env.done is True
    assert env.reward == first.progress_reward - 1.0


def test_truncated_response_preserves_progress_and_applies_penalty() -> None:
    env = make_env(TWO_MOVES, 2)
    first = env._move(7)

    result = env._fail("truncated")

    assert result.status == "truncated"
    assert env.done is True
    assert env.reward == first.progress_reward - 1.0


def test_malformed_response_is_terminal() -> None:
    env = make_env(ONE_MOVE, 1)

    result = env._fail("malformed")

    assert result.status == "malformed"
    assert env.done is True
    assert env.reward == -1.0


def test_environment_tool_returns_observation_and_raises_for_illegal_move() -> None:
    env = make_env(ONE_MOVE, 1)

    assert "Legal tiles: [5, 7, 8]" in env._observation()
    with pytest.raises(ValueError, match="illegal tile 1"):
        env.slide_tile(1)
    assert env.get_reward() == -1.0


def test_environment_rejects_wrong_optimal_length() -> None:
    with pytest.raises(ValueError, match="exact solution distance"):
        make_env(ONE_MOVE, 2)


def test_environment_requires_reset_before_use() -> None:
    env = PuzzleEnv()

    with pytest.raises(RuntimeError, match="reset"):
        env._move(8)
