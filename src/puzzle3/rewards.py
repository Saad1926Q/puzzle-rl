"""Reward primitives shared by evaluation and RL training."""

from __future__ import annotations

from puzzle3.board import Board
from puzzle3.solver import exact_distance

SOLVED_BASE_REWARD = 0.8
SOLVED_EFFICIENCY_WEIGHT = 0.2
DISTANCE_PROGRESS_WEIGHT = 0.5
MAX_PUZZLE_DISTANCE = 31


def solved_reward(optimal_length: int, moves_taken: int) -> float:
    """Return the bounded reward for a solved trajectory."""

    if moves_taken <= 0:
        return 1.0
    efficiency = min(optimal_length / moves_taken, 1.0)
    return SOLVED_BASE_REWARD + SOLVED_EFFICIENCY_WEIGHT * efficiency


def distance_progress_reward(
    board: Board,
    next_board: Board,
    *,
    weight: float = DISTANCE_PROGRESS_WEIGHT,
) -> float:
    """Reward a transition according to its exact-distance improvement."""

    distance_improvement = exact_distance(board) - exact_distance(next_board)
    return weight * distance_improvement / MAX_PUZZLE_DISTANCE
