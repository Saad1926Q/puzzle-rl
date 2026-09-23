"""Authoritative 8-puzzle rollout and reward evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable

from tqdm import tqdm

from evaluation.constants import DEFAULT_HISTORY_TURNS, DEFAULT_MAX_TURNS, MAX_TURNS
from evaluation.dataset import PuzzleExample
from evaluation.protocol import HistoryTurn, PuzzleAgent, parse_tile
from evaluation.results import EpisodeResult, EvaluationResult, StepResult
from evaluation.rewards import distance_progress_reward, episode_reward, solved_reward
from puzzle3.environment import PuzzleEnv

__all__ = [
    "distance_progress_reward",
    "episode_reward",
    "evaluate",
    "evaluate_episode",
    "EvaluationResult",
    "EpisodeResult",
    "solved_reward",
    "StepResult",
]


def _discard_progress_rewards(steps: list[StepResult]) -> None:
    """Keep progress diagnostics while removing them from the episode return."""

    for step in steps:
        step.reward = 0.0


def evaluate_episode(
    example: PuzzleExample,
    agent: PuzzleAgent,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    rollout_id: int = 0,
    keep_history: bool = False,
) -> EpisodeResult:
    """Run one puzzle through the shared environment."""

    if not 1 <= max_turns <= MAX_TURNS:
        raise ValueError(f"max_turns must be between 1 and {MAX_TURNS}")

    environment = PuzzleEnv()
    environment.reset(
        board=example.board,
        optimal_length=example.optimal_length,
        max_turns=max_turns,
    )
    history: list[HistoryTurn] = []
    steps: list[StepResult] = []

    def record_step(
        turn: int,
        result: Any,
        raw_response: str | None,
        response_metadata: dict[str, Any] | None,
    ) -> None:
        if result.status in {"solved", "illegal", "malformed", "truncated"}:
            _discard_progress_rewards(steps)
        steps.append(
            StepResult(
                turn=turn,
                board=result.board,
                legal_tiles=result.legal_tiles,
                raw_response=raw_response,
                tile=result.tile,
                next_board=result.next_board,
                status=result.status,
                response_metadata=response_metadata,
                reward=result.reward,
                progress_reward=result.progress_reward,
                terminal_reward=result.terminal_reward,
            )
        )

    while not environment.done:
        board = environment.board
        assert board is not None
        turn = environment.moves + 1
        try:
            raw_response = (
                agent.next_action(
                    board,
                    tuple(history[-DEFAULT_HISTORY_TURNS:]),
                    include_reasoning=True,
                )
                if keep_history
                else agent.next_action(board)
            )
        except Exception:
            response_metadata = getattr(agent, "last_response_metadata", None)
            if not response_metadata or response_metadata.get("status") != "api_error":
                raise
            result = environment._fail("api_error")
            record_step(turn, result, None, response_metadata)
            break

        response_metadata = getattr(agent, "last_response_metadata", None)
        tile = parse_tile(raw_response)
        if tile is None:
            status = (
                "truncated"
                if response_metadata and response_metadata.get("truncated")
                else "malformed"
            )
            result = environment._fail(status)
            record_step(turn, result, raw_response, response_metadata)
            break

        result = environment._move(tile)
        record_step(turn, result, raw_response, response_metadata)
        if result.done:
            break

        history.append(
            HistoryTurn(
                board=result.board,
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

    final_board = environment.board
    assert final_board is not None
    return EpisodeResult(
        example=example,
        outcome=environment.outcome,
        reward=episode_reward(steps) if steps else environment.reward,
        moves_taken=environment.moves,
        final_board=final_board,
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
    agent_factory: Callable[[], PuzzleAgent] | None = None,
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
            keep_history=keep_history,
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
