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
    DISTANCE_PROGRESS_WEIGHT,
    MAX_PUZZLE_DISTANCE,
    SOLVED_BASE_REWARD,
    SOLVED_EFFICIENCY_WEIGHT,
    TIMEOUT_REWARD,
)
from evaluation.dataset import PuzzleExample
from evaluation.protocol import HistoryTurn, PuzzleAgent, parse_tile
from puzzle3.board import Board, TileAction, adjacent_tiles, is_solved, slide_tile
from puzzle3.solver import exact_distance


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


def _episode_reward(steps: list[StepResult]) -> float:
    """Aggregate per-transition rewards into the episode return."""

    return sum(step.reward for step in steps)


def _failed_episode(
    *,
    example: PuzzleExample,
    rollout_id: int,
    turn: int,
    board: Board,
    legal_tiles: tuple[TileAction, ...],
    raw_response: str | None,
    tile: TileAction | None,
    outcome: str,
    response_metadata: dict[str, Any] | None,
    steps: list[StepResult],
    terminal_reward: float = ILLEGAL_OR_MALFORMED_REWARD,
) -> EpisodeResult:
    steps.append(
        StepResult(
            turn=turn,
            board=board,
            legal_tiles=legal_tiles,
            raw_response=raw_response,
            tile=tile,
            next_board=None,
            status=outcome,
            response_metadata=response_metadata,
            reward=terminal_reward,
            terminal_reward=terminal_reward,
        )
    )
    return EpisodeResult(
        example=example,
        outcome=outcome,
        reward=_episode_reward(steps),
        moves_taken=turn - 1,
        final_board=board,
        steps=steps,
        rollout_id=rollout_id,
    )


def evaluate_episode(
    example: PuzzleExample,
    agent: PuzzleAgent,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    rollout_id: int = 0,
    keep_history: bool = False,
    keep_reasoning: bool = False,
) -> EpisodeResult:
    """Run one puzzle with environment-authoritative distance-progress scoring.

    Invalid responses receive only the terminal penalty because they do not produce
    a valid successor state. Every valid move is rewarded for exact-distance progress;
    solving and timeout rewards are attached to the terminal transition so that the
    per-step rewards sum exactly to the episode return.
    """

    if not 1 <= max_turns <= MAX_TURNS:
        raise ValueError(f"max_turns must be between 1 and {MAX_TURNS}")
    if keep_reasoning and not keep_history:
        raise ValueError("keep_reasoning requires keep_history")

    board = example.board
    history: list[HistoryTurn] = []
    steps: list[StepResult] = []
    if is_solved(board):
        return EpisodeResult(
            example=example,
            outcome="solved",
            reward=solved_reward(example.optimal_length, 0),
            moves_taken=0,
            final_board=board,
            steps=steps,
            rollout_id=rollout_id,
        )

    for turn in range(1, max_turns + 1):
        available = adjacent_tiles(board)
        try:
            raw_response = (
                agent.next_action(
                    board,
                    history[-4:],
                    include_reasoning=keep_reasoning,
                )
                if keep_history
                else agent.next_action(board)
            )
        except Exception:
            response_metadata = getattr(agent, "last_response_metadata", None)
            if not response_metadata or response_metadata.get("status") != "api_error":
                raise
            return _failed_episode(
                example=example,
                rollout_id=rollout_id,
                turn=turn,
                board=board,
                legal_tiles=available,
                raw_response=None,
                tile=None,
                outcome="api_error",
                response_metadata=response_metadata,
                steps=steps,
                terminal_reward=0.0,
            )
        response_metadata = getattr(agent, "last_response_metadata", None)
        tile = parse_tile(raw_response)
        if tile is None:
            outcome = (
                "truncated"
                if response_metadata and response_metadata.get("truncated")
                else "malformed"
            )
            return _failed_episode(
                example=example,
                rollout_id=rollout_id,
                turn=turn,
                board=board,
                legal_tiles=available,
                raw_response=raw_response,
                tile=None,
                outcome=outcome,
                response_metadata=response_metadata,
                steps=steps,
            )

        if tile not in available:
            return _failed_episode(
                example=example,
                rollout_id=rollout_id,
                turn=turn,
                board=board,
                legal_tiles=available,
                raw_response=raw_response,
                tile=tile,
                outcome="illegal",
                response_metadata=response_metadata,
                steps=steps,
            )

        next_board = slide_tile(board, tile)
        solved = is_solved(next_board)
        progress_reward = distance_progress_reward(board, next_board)
        terminal_reward = solved_reward(example.optimal_length, turn) if solved else 0.0
        steps.append(
            StepResult(
                turn=turn,
                board=board,
                legal_tiles=available,
                raw_response=raw_response,
                tile=tile,
                next_board=next_board,
                status="solved" if solved else "valid",
                response_metadata=response_metadata,
                reward=progress_reward + terminal_reward,
                progress_reward=progress_reward,
                terminal_reward=terminal_reward,
            )
        )
        history.append(
            HistoryTurn(
                board=board,
                tile=tile,
                reasoning=(
                    response_metadata.get("reasoning_content", "")
                    if response_metadata
                    else ""
                ),
                reasoning_details=(
                    response_metadata.get("reasoning_details")
                    if response_metadata
                    else None
                ),
            )
        )
        board = next_board
        if solved:
            return EpisodeResult(
                example=example,
                outcome="solved",
                reward=_episode_reward(steps),
                moves_taken=turn,
                final_board=board,
                steps=steps,
                rollout_id=rollout_id,
            )

    # Attach the timeout penalty to the final valid transition. This keeps the
    # serialized per-step rewards aligned with the aggregate episode return.
    steps[-1].reward += TIMEOUT_REWARD
    steps[-1].terminal_reward += TIMEOUT_REWARD
    return EpisodeResult(
        example=example,
        outcome="timeout",
        reward=_episode_reward(steps),
        moves_taken=max_turns,
        final_board=board,
        steps=steps,
        rollout_id=rollout_id,
    )


def evaluate(
    examples: Iterable[PuzzleExample],
    agent: PuzzleAgent | None = None,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    num_rollouts: int = 1,
    parallelism: int = 1,
    keep_history: bool = False,
    keep_reasoning: bool = False,
    agent_factory: Callable[[], PuzzleAgent] | None = None,
) -> EvaluationResult:
    """Evaluate independent puzzle rollouts, optionally in parallel.

    Turns within one episode remain sequential. When ``parallelism > 1``, callers
    must provide ``agent_factory`` so each worker gets an isolated model client and
    response metadata store; sharing one mutable agent across threads is unsafe.
    """

    if keep_reasoning and not keep_history:
        raise ValueError("keep_reasoning requires keep_history")
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
            keep_history=keep_history,
            keep_reasoning=keep_reasoning,
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
