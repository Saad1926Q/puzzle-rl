"""Authoritative 8-puzzle rollout and reward evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable

from tqdm import tqdm

from evaluation.constants import (
    DEFAULT_MAX_TURNS,
    ILLEGAL_OR_MALFORMED_REWARD,
    MAX_TURNS,
    TIMEOUT_REWARD,
)
from evaluation.dataset import PuzzleExample
from evaluation.protocol import HistoryTurn, PuzzleAgent, parse_tile
from evaluation.results import EpisodeResult, EvaluationResult, StepResult
from evaluation.rewards import (
    distance_progress_reward,
    episode_reward,
    solved_reward,
)
from puzzle3.board import Board, TileAction, adjacent_tiles, is_solved, slide_tile




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
        reward=episode_reward(steps),
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
                reward=episode_reward(steps),
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
        reward=episode_reward(steps),
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
