from __future__ import annotations

import pytest

from puzzle3.board import GOAL
from puzzle3.episode import PuzzleEpisode
from puzzle3.rewards import distance_progress_reward, solved_reward


ONE_MOVE = (1, 2, 3, 4, 5, 6, 7, 0, 8)
TWO_MOVES = (1, 2, 3, 4, 5, 6, 0, 7, 8)


def test_episode_solves_and_uses_terminal_reward_only() -> None:
    episode = PuzzleEpisode(ONE_MOVE, optimal_length=1)

    transition = episode.step(8)

    assert transition.status == "solved"
    assert transition.done is True
    assert episode.board == GOAL
    assert episode.moves_taken == 1
    assert episode.reward == solved_reward(1, 1)
    assert episode.progress_reward == distance_progress_reward(ONE_MOVE, GOAL)


def test_episode_timeout_keeps_accumulated_progress() -> None:
    episode = PuzzleEpisode(TWO_MOVES, optimal_length=2, max_turns=1)

    transition = episode.step(7)

    assert transition.status == "timeout"
    assert episode.done is True
    assert episode.reward == transition.progress_reward
    assert episode.progress_reward == transition.progress_reward


def test_illegal_action_discards_previous_progress_from_return() -> None:
    episode = PuzzleEpisode(TWO_MOVES, optimal_length=2, max_turns=3)
    episode.step(7)

    transition = episode.step(1)

    assert transition.status == "illegal"
    assert episode.outcome == "illegal"
    assert episode.reward == -1.0
    assert episode.progress_reward > 0.0


def test_non_action_failure_is_terminal() -> None:
    episode = PuzzleEpisode(ONE_MOVE, optimal_length=1)

    transition = episode.fail("malformed")

    assert transition.status == "malformed"
    assert transition.done is True
    assert episode.reward == -1.0


@pytest.mark.parametrize("board,optimal_length", [(GOAL, 0), (ONE_MOVE, 1)])
def test_episode_initializes_valid_board(board: tuple[int, ...], optimal_length: int) -> None:
    episode = PuzzleEpisode(board, optimal_length)

    assert episode.board == board
    assert episode.done is (board == GOAL)


def test_episode_rejects_incorrect_optimal_length() -> None:
    with pytest.raises(ValueError, match="exact solution distance"):
        PuzzleEpisode(ONE_MOVE, optimal_length=2)
