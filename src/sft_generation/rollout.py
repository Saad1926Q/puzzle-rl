"""Generate and validate successful 8-puzzle teacher trajectories."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from evaluation.clients.openrouter import OpenRouterAgent
from evaluation.constants import DEFAULT_OPENROUTER_BASE_URL
from evaluation.dataset import PuzzleExample
from evaluation.evaluator import EpisodeResult, evaluate_episode
from evaluation.protocol import get_api_key
from puzzle3.board import Board, GOAL, adjacent_tiles, is_solved, slide_tile
from sft_generation.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TURNS,
    DEFAULT_TEACHER_MODEL,
    DEFAULT_THINKING,
    SFT_SOLVING_PROMPT_WITH_HISTORY,
)
@dataclass(frozen=True)
class RolloutFailure:
    """Record one rollout exception without losing the batch."""

    source_id: str
    rollout_id: int
    error: str


@dataclass(frozen=True)
class RolloutConfig:
    """Settings shared by all rollouts in one generation run."""

    model: str = DEFAULT_TEACHER_MODEL
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    max_turns: int = DEFAULT_MAX_TURNS
    max_tokens: int = DEFAULT_MAX_TOKENS
    thinking: bool = DEFAULT_THINKING
    reasoning_effort: str | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    upstream_providers: tuple[str, ...] = ()
    allow_fallbacks: bool = False
    require_parameters: bool = True
    data_collection: str = "deny"
    provider_retries: int = 2
    retry_delay: float = 1.0


def run_rollout(
    example: PuzzleExample,
    *,
    rollout_id: int,
    api_key: str,
    config: RolloutConfig,
    client_factory: Callable[..., OpenRouterAgent] = OpenRouterAgent,
) -> EpisodeResult:
    """Run one independent teacher rollout."""

    agent = client_factory(
        api_key=api_key,
        model=config.model,
        base_url=config.base_url,
        thinking=config.thinking,
        reasoning_effort=config.reasoning_effort,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        upstream_providers=config.upstream_providers,
        allow_fallbacks=config.allow_fallbacks,
        require_parameters=config.require_parameters,
        data_collection=config.data_collection,
        provider_retries=config.provider_retries,
        retry_delay=config.retry_delay,
        system_prompt=SFT_SOLVING_PROMPT_WITH_HISTORY,
    )
    return evaluate_episode(
        example,
        agent,
        max_turns=config.max_turns,
        rollout_id=rollout_id,
        keep_history=True,
        keep_reasoning=True,
    )


def select_successful_rollout(episodes: Iterable[EpisodeResult]) -> EpisodeResult | None:
    """Select the lowest-numbered solved rollout."""

    successful = [episode for episode in episodes if episode.solved]
    return min(successful, key=lambda episode: episode.rollout_id, default=None)


def trajectory_record(episode: EpisodeResult) -> dict[str, Any]:
    """Convert a solved episode to the SFT trajectory schema."""

    if not episode.solved:
        raise ValueError("only solved episodes can become trajectories")
    record = {
        "source_id": episode.example.example_id,
        "initial_board": list(episode.example.board),
        "optimal_length": episode.example.optimal_length,
        "rollout_id": episode.rollout_id,
        "moves_taken": episode.moves_taken,
        "final_board": list(episode.final_board),
        "teacher_model": None,
        "steps": [],
    }
    for step in episode.steps:
        reasoning = ""
        if step.response_metadata:
            reasoning = step.response_metadata.get("reasoning_content", "") or ""
        record["steps"].append(
            {
                "turn": step.turn,
                "board": list(step.board),
                "legal_tiles": list(step.legal_tiles),
                "tile": step.tile,
                "next_board": list(step.next_board) if step.next_board else None,
                "status": step.status,
                "raw_response": step.raw_response,
                "reasoning": reasoning,
                "response_metadata": step.response_metadata,
            }
        )
    validate_trajectory(record)
    return record


def validate_trajectory(record: dict[str, Any]) -> None:
    """Replay a trajectory and reject any inconsistent transition."""

    board = _board(record.get("initial_board"), "initial_board")
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("trajectory must contain at least one step")
    for expected_turn, step in enumerate(steps, start=1):
        if not isinstance(step, dict) or step.get("turn") != expected_turn:
            raise ValueError("trajectory turns must be contiguous")
        if _board(step.get("board"), "step board") != board:
            raise ValueError(f"turn {expected_turn} has an incorrect board")
        tile = step.get("tile")
        if type(tile) is not int or tile not in adjacent_tiles(board):
            raise ValueError(f"turn {expected_turn} has an illegal tile")
        next_board = slide_tile(board, tile)
        if _board(step.get("next_board"), "next_board") != next_board:
            raise ValueError(f"turn {expected_turn} has an incorrect next board")
        board = next_board
    if not is_solved(board) or board != GOAL:
        raise ValueError("trajectory does not end at the goal")
    if record.get("moves_taken") != len(steps):
        raise ValueError("moves_taken does not match the step count")
    if _board(record.get("final_board"), "final_board") != board:
        raise ValueError("final_board does not match the replay")


def _board(value: Any, field: str) -> Board:
    """Validate and convert one serialized board."""

    if not isinstance(value, list | tuple) or len(value) != 9:
        raise ValueError(f"{field} must contain nine integers")
    board = tuple(value)
    if any(type(tile) is not int for tile in board) or set(board) != set(GOAL):
        raise ValueError(f"{field} must be a permutation of 0 through 8")
    return board


def rollout_futures(
    examples: Iterable[PuzzleExample],
    *,
    num_rollouts: int,
    parallelism: int,
    api_key: str,
    config: RolloutConfig,
) -> dict[str, list[EpisodeResult | RolloutFailure]]:
    """Run all requested rollouts and group results by source ID."""

    if num_rollouts < 1 or parallelism < 1:
        raise ValueError("num_rollouts and parallelism must be positive")
    grouped: dict[str, list[EpisodeResult | RolloutFailure]] = {}
    example_list = list(examples)
    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures: dict[Future[EpisodeResult], tuple[str, int]] = {}
        for example in example_list:
            grouped[example.example_id] = []
            for rollout_id in range(1, num_rollouts + 1):
                future = pool.submit(
                    run_rollout,
                    example,
                    rollout_id=rollout_id,
                    api_key=api_key,
                    config=config,
                )
                futures[future] = (example.example_id, rollout_id)
        for future in as_completed(futures):
            source_id, rollout_id = futures[future]
            try:
                grouped[source_id].append(future.result())
            except Exception as exc:
                grouped[source_id].append(
                    RolloutFailure(source_id, rollout_id, str(exc))
                )
    return grouped




def api_key_from_environment(env_name: str) -> str:
    """Load the configured API key."""

    return get_api_key(env_name)
