"""Authoritative 8-puzzle rollout and reward evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from tqdm import tqdm

from evaluation.constants import (
    DEFAULT_MAX_TURNS,
    ILLEGAL_OR_MALFORMED_REWARD,
    MAX_TURNS,
    SOLVED_BASE_REWARD,
    SOLVED_EFFICIENCY_WEIGHT,
    TIMEOUT_REWARD,
)
from evaluation.dataset import PuzzleExample
from evaluation.protocol import MoveAgent, parse_move
from puzzle3.board import apply_move, is_solved, legal_moves


@dataclass
class StepResult:
    turn: int
    board: tuple[int, ...]
    legal_moves: tuple[str, ...]
    raw_response: str | None
    move: str | None
    next_board: tuple[int, ...] | None
    status: str
    response_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "board": list(self.board),
            "legal_moves": list(self.legal_moves),
            "raw_response": self.raw_response,
            "move": self.move,
            "next_board": list(self.next_board) if self.next_board is not None else None,
            "status": self.status,
            "response_metadata": self.response_metadata,
        }


@dataclass
class EpisodeResult:
    example: PuzzleExample
    outcome: str
    reward: float
    moves_taken: int
    final_board: tuple[int, ...]
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
        outcomes = {outcome: sum(e.outcome == outcome for e in self.episodes) for outcome in (
            "solved", "illegal", "malformed", "truncated", "timeout"
        )}
        return {
            "num_examples": num_examples,
            "num_rollouts": self.num_rollouts,
            "num_episodes": count,
            "solved": outcomes["solved"],
            "illegal": outcomes["illegal"],
            "malformed": outcomes["malformed"],
            "truncated": outcomes["truncated"],
            "timeout": outcomes["timeout"],
            "solved_rate": outcomes["solved"] / count if count else 0.0,
            "illegal_rate": outcomes["illegal"] / count if count else 0.0,
            "malformed_rate": outcomes["malformed"] / count if count else 0.0,
            "truncated_rate": outcomes["truncated"] / count if count else 0.0,
            "timeout_rate": outcomes["timeout"] / count if count else 0.0,
            # pass@k is the fraction of distinct puzzles solved by at least one
            # of the k independent rollouts.
            "pass@k": len(solved_examples) / num_examples if num_examples else 0.0,
            "mean_reward": sum(e.reward for e in self.episodes) / count if count else 0.0,
            "mean_moves_taken": sum(e.moves_taken for e in self.episodes) / count if count else 0.0,
            "mean_solved_moves": (
                sum(e.moves_taken for e in solved) / len(solved) if solved else 0.0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary(), "episodes": [e.to_dict() for e in self.episodes]}


def solved_reward(optimal_length: int, moves_taken: int) -> float:
    """Return the exact requested solved-trajectory reward."""

    if moves_taken <= 0:
        return 1.0
    efficiency = min(optimal_length / moves_taken, 1.0)
    return SOLVED_BASE_REWARD + SOLVED_EFFICIENCY_WEIGHT * efficiency


def evaluate_episode(
    example: PuzzleExample,
    agent: MoveAgent,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    rollout_id: int = 0,
) -> EpisodeResult:
    """Run one puzzle with environment-authoritative state and scoring."""

    if max_turns <= 0 or max_turns > MAX_TURNS:
        raise ValueError(f"max_turns must be between 1 and {MAX_TURNS}")

    board = example.board
    steps: list[StepResult] = []
    if is_solved(board):
        return EpisodeResult(example, "solved", 1.0, 0, board, steps, rollout_id)

    for turn in range(1, max_turns + 1):
        available = tuple(legal_moves(board))
        raw_response = agent.next_move(board)
        response_metadata = getattr(agent, "last_response_metadata", None)
        move = parse_move(raw_response)
        if move is None:
            outcome = (
                "truncated"
                if response_metadata and response_metadata.get("truncated")
                else "malformed"
            )
            steps.append(
                StepResult(
                    turn, board, available, raw_response, None, None, outcome,
                    response_metadata,
                )
            )
            return EpisodeResult(
                example, outcome, ILLEGAL_OR_MALFORMED_REWARD, turn - 1, board, steps,
                rollout_id,
            )
        if move not in available:
            steps.append(
                StepResult(
                    turn, board, available, raw_response, move, None, "illegal",
                    response_metadata,
                )
            )
            return EpisodeResult(
                example, "illegal", ILLEGAL_OR_MALFORMED_REWARD, turn - 1, board, steps,
                rollout_id,
            )

        next_board = apply_move(board, move)
        solved = is_solved(next_board)
        steps.append(
            StepResult(
                turn,
                board,
                available,
                raw_response,
                move,
                next_board,
                "solved" if solved else "valid",
                response_metadata,
            )
        )
        board = next_board
        if solved:
            return EpisodeResult(
                example,
                "solved",
                solved_reward(example.optimal_length, turn),
                turn,
                board,
                steps,
                rollout_id,
            )

    return EpisodeResult(
        example, "timeout", TIMEOUT_REWARD, max_turns, board, steps, rollout_id
    )


def evaluate(
    examples: Iterable[PuzzleExample],
    agent: MoveAgent | None = None,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    num_rollouts: int = 1,
    parallelism: int = 1,
    agent_factory: Callable[[], MoveAgent] | None = None,
) -> EvaluationResult:
    """Evaluate independent puzzle rollouts, optionally in parallel.

    Turns within one episode remain sequential. When ``parallelism > 1``, callers
    must provide ``agent_factory`` so each worker gets an isolated model client and
    response metadata store; sharing one mutable agent across threads is unsafe.
    """

    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive")
    if parallelism <= 0:
        raise ValueError("parallelism must be positive")
    if agent is None and agent_factory is None:
        raise ValueError("provide agent or agent_factory")
    if parallelism > 1 and agent_factory is None:
        raise ValueError("agent_factory is required when parallelism > 1")

    examples = list(examples)
    jobs = [
        (example_index, example, rollout_id)
        for example_index, example in enumerate(examples)
        for rollout_id in range(num_rollouts)
    ]

    def run_job(job: tuple[int, PuzzleExample, int]) -> tuple[int, int, EpisodeResult]:
        example_index, example, rollout_id = job
        episode_agent = agent_factory() if agent_factory is not None else agent
        assert episode_agent is not None
        episode = evaluate_episode(
            example,
            episode_agent,
            max_turns=max_turns,
            rollout_id=rollout_id,
        )
        return example_index, rollout_id, episode

    episodes_by_key: dict[tuple[int, int], EpisodeResult] = {}
    progress = tqdm(total=len(jobs), desc="Evaluating", unit="episode")
    try:
        if parallelism == 1:
            for job in jobs:
                example_index, rollout_id, episode = run_job(job)
                episodes_by_key[(example_index, rollout_id)] = episode
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                futures = [executor.submit(run_job, job) for job in jobs]
                for future in as_completed(futures):
                    example_index, rollout_id, episode = future.result()
                    episodes_by_key[(example_index, rollout_id)] = episode
                    progress.update(1)
    finally:
        progress.close()

    episodes = [
        episodes_by_key[(example_index, rollout_id)]
        for example_index, _example in enumerate(examples)
        for rollout_id in range(num_rollouts)
    ]
    return EvaluationResult(episodes, num_rollouts=num_rollouts)
