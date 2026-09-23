"""Reward functions for authoritative 8-puzzle evaluation."""

from __future__ import annotations

from evaluation.results import StepResult
from puzzle3.rewards import distance_progress_reward, solved_reward

__all__ = ["distance_progress_reward", "episode_reward", "solved_reward"]


def episode_reward(steps: list[StepResult]) -> float:
    """Aggregate per-transition rewards into the episode return."""

    return sum(step.reward for step in steps)
