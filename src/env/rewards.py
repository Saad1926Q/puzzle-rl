"""Reward helpers for puzzle episodes."""

from __future__ import annotations

from typing import Any

SOLVED_BASE = 0.8
SOLVED_EFFICIENCY_WEIGHT = 0.2
PENALTY = -0.1

METRIC_KEYS = ("solved", "illegal_move", "format_failure", "num_moves", "efficiency")


def final_reward(
    *,
    solved: bool,
    moves_taken: int,
    optimal_length: int,
    illegal_move: bool = False,
    format_failure: bool = False,
) -> float:
    """Score a finished episode."""

    if solved:
        efficiency = optimal_length / moves_taken
        return SOLVED_BASE + SOLVED_EFFICIENCY_WEIGHT * min(efficiency, 1.0)
    if illegal_move or format_failure:
        return PENALTY
    return 0.0


def metrics(
    *,
    solved: bool,
    illegal_move: bool,
    format_failure: bool,
    moves_taken: int,
    optimal_length: int,
) -> dict[str, float]:
    """Return scalar episode metrics."""

    efficiency = optimal_length / moves_taken if solved and moves_taken else 0.0
    return {
        "solved": float(solved),
        "illegal_move": float(illegal_move),
        "format_failure": float(format_failure),
        "num_moves": float(moves_taken),
        "efficiency": efficiency,
    }


def flat_metrics(metrics_dict: dict[str, Any]) -> dict[str, float]:
    """Keep only scalar metrics for logging."""

    out: dict[str, float] = {}
    for key, value in metrics_dict.items():
        if isinstance(value, bool):
            out[key] = float(value)
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out
