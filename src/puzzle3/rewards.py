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
    initial_distance: int,
    weight: float = DISTANCE_PROGRESS_WEIGHT,
) -> float:
    """Reward one transition by its initial-distance-normalized progress."""

    if initial_distance <= 0:
        raise ValueError("initial_distance must be positive")
    distance_improvement = exact_distance(board) - exact_distance(next_board)
    return weight * distance_improvement / initial_distance


def bounded_progress_reward(
    initial_distance: int,
    current_distance: int,
    *,
    weight: float = DISTANCE_PROGRESS_WEIGHT,
) -> float:
    """Return bounded net progress relative to the initial puzzle distance."""

    if initial_distance < 0:
        raise ValueError("initial_distance must be non-negative")
    if initial_distance == 0:
        return 0.0
    normalized_progress = (initial_distance - current_distance) / initial_distance
    bounded_progress = max(-1.0, min(1.0, normalized_progress))
    return weight * bounded_progress
