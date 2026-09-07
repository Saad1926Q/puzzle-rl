"""Reward functions for authoritative 8-puzzle evaluation."""

from __future__ import annotations

from evaluation.constants import (
    DISTANCE_PROGRESS_WEIGHT,
    MAX_PUZZLE_DISTANCE,
    SOLVED_BASE_REWARD,
    SOLVED_EFFICIENCY_WEIGHT,
)
from evaluation.results import StepResult
from puzzle3.board import Board
from puzzle3.solver import exact_distance


def solved_reward(optimal_length: int, moves_taken: int) -> float:
    """Return the exact requested solved-trajectory reward."""

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
    """Reward a valid transition according to its exact-distance improvement."""

    distance_improvement = exact_distance(board) - exact_distance(next_board)
    return weight * distance_improvement / MAX_PUZZLE_DISTANCE


def episode_reward(steps: list[StepResult]) -> float:
    """Aggregate per-transition rewards into the episode return."""

    return sum(step.reward for step in steps)
