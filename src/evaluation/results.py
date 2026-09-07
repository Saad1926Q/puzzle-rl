"""Serializable results and aggregate metrics for puzzle evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.dataset import PuzzleExample
from puzzle3.board import Board, TileAction


@dataclass
class StepResult:
    turn: int
    board: Board
    legal_tiles: tuple[TileAction, ...]
    raw_response: str | None
    tile: TileAction | None
    next_board: Board | None
    status: str
    response_metadata: dict[str, Any] | None = None
    reward: float = 0.0
    progress_reward: float = 0.0
    terminal_reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "board": list(self.board),
            "legal_tiles": list(self.legal_tiles),
            "raw_response": self.raw_response,
            "tile": self.tile,
            "next_board": list(self.next_board)
            if self.next_board is not None
            else None,
            "status": self.status,
            "response_metadata": self.response_metadata,
            "reward": self.reward,
            "progress_reward": self.progress_reward,
            "terminal_reward": self.terminal_reward,
        }


@dataclass
class EpisodeResult:
    example: PuzzleExample
    outcome: str
    reward: float
    moves_taken: int
    final_board: Board
    steps: list[StepResult]
    rollout_id: int = 0

    @property
    def solved(self) -> bool:
        return self.outcome == "solved"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.example.example_id,
            "rollout_id": self.rollout_id,
            "initial_board": list(self.example.board),
            "optimal_length": self.example.optimal_length,
            "outcome": self.outcome,
            "reward": self.reward,
            "moves_taken": self.moves_taken,
            "final_board": list(self.final_board),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class EvaluationResult:
    episodes: list[EpisodeResult]
    num_rollouts: int = 1

    def summary(self) -> dict[str, Any]:
        examples = {episode.example.example_id for episode in self.episodes}
        count = len(self.episodes)
        num_examples = len(examples)
        solved = [episode for episode in self.episodes if episode.solved]
        solved_examples = {
            episode.example.example_id for episode in self.episodes if episode.solved
        }
        outcomes = {
            outcome: sum(e.outcome == outcome for e in self.episodes)
            for outcome in (
                "solved",
                "illegal",
                "malformed",
                "truncated",
                "timeout",
                "api_error",
            )
        }
        return {
            "num_examples": num_examples,
            "num_rollouts": self.num_rollouts,
            "num_episodes": count,
            "solved": outcomes["solved"],
            "illegal": outcomes["illegal"],
            "malformed": outcomes["malformed"],
            "truncated": outcomes["truncated"],
            "timeout": outcomes["timeout"],
            "api_error": outcomes["api_error"],
            "solved_rate": outcomes["solved"] / count if count else 0.0,
            "illegal_rate": outcomes["illegal"] / count if count else 0.0,
            "malformed_rate": outcomes["malformed"] / count if count else 0.0,
            "truncated_rate": outcomes["truncated"] / count if count else 0.0,
            "timeout_rate": outcomes["timeout"] / count if count else 0.0,
            "api_error_rate": outcomes["api_error"] / count if count else 0.0,
            # pass@k is the fraction of distinct puzzles solved by at least one
            # of the k independent rollouts.
            "pass@k": len(solved_examples) / num_examples if num_examples else 0.0,
            "mean_reward": sum(e.reward for e in self.episodes) / count
            if count
            else 0.0,
            "mean_moves_taken": sum(e.moves_taken for e in self.episodes) / count
            if count
            else 0.0,
            "mean_solved_moves": (
                sum(e.moves_taken for e in solved) / len(solved) if solved else 0.0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "episodes": [e.to_dict() for e in self.episodes],
        }
