"""Authoritative puzzle rollout owned by prompt optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.dataset import PuzzleExample
from prompt_optimization.eval.constants import (
    DISTANCE_PROGRESS_WEIGHT,
    ILLEGAL_OR_MALFORMED_REWARD,
    MAX_PUZZLE_DISTANCE,
    MAX_TURNS,
    SOLVED_BASE_REWARD,
    SOLVED_EFFICIENCY_WEIGHT,
    TIMEOUT_REWARD,
)
from prompt_optimization.eval.protocol import HistoryTurn, PuzzleAgent, parse_tile
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


@dataclass
class EpisodeResult:
    example: PuzzleExample
    outcome: str
    reward: float
    moves_taken: int
    final_board: Board
    steps: list[StepResult]
    rollout_id: int = 0


def solved_reward(optimal_length: int, moves_taken: int) -> float:
    if moves_taken <= 0:
        return 1.0
    return SOLVED_BASE_REWARD + SOLVED_EFFICIENCY_WEIGHT * min(optimal_length / moves_taken, 1.0)


def distance_progress_reward(board: Board, next_board: Board) -> float:
    return DISTANCE_PROGRESS_WEIGHT * (exact_distance(board) - exact_distance(next_board)) / MAX_PUZZLE_DISTANCE


def _failed(
    example: PuzzleExample,
    turn: int,
    board: Board,
    legal_tiles: tuple[TileAction, ...],
    raw_response: str | None,
    tile: TileAction | None,
    outcome: str,
    metadata: dict[str, Any] | None,
    steps: list[StepResult],
    terminal_reward: float = ILLEGAL_OR_MALFORMED_REWARD,
) -> EpisodeResult:
    steps.append(StepResult(
        turn=turn,
        board=board,
        legal_tiles=legal_tiles,
        raw_response=raw_response,
        tile=tile,
        next_board=None,
        status=outcome,
        response_metadata=metadata,
        reward=terminal_reward,
        terminal_reward=terminal_reward,
    ))
    return EpisodeResult(
        example,
        outcome,
        sum(step.reward for step in steps),
        turn - 1,
        board,
        steps,
    )


def evaluate_episode(
    example: PuzzleExample,
    agent: PuzzleAgent,
    *,
    max_turns: int = MAX_TURNS,
    keep_history: bool = False,
    keep_reasoning: bool = False,
) -> EpisodeResult:
    """Run one candidate prompt against environment-authoritative transitions."""

    if not 1 <= max_turns <= MAX_TURNS:
        raise ValueError(f"max_turns must be between 1 and {MAX_TURNS}")
    if keep_reasoning and not keep_history:
        raise ValueError("keep_reasoning requires keep_history")
    board = example.board
    history: list[HistoryTurn] = []
    steps: list[StepResult] = []
    if is_solved(board):
        return EpisodeResult(example, "solved", solved_reward(example.optimal_length, 0), 0, board, steps)

    for turn in range(1, max_turns + 1):
        legal_tiles = adjacent_tiles(board)
        try:
            raw_response = agent.next_action(board, history[-4:], include_reasoning=keep_reasoning) if keep_history else agent.next_action(board)
        except Exception:
            metadata = getattr(agent, "last_response_metadata", None)
            if not metadata or metadata.get("status") != "api_error":
                raise
            return _failed(example, turn, board, legal_tiles, None, None, "api_error", metadata, steps, 0.0)
        metadata = getattr(agent, "last_response_metadata", None)
        tile = parse_tile(raw_response)
        if tile is None:
            outcome = "truncated" if metadata and metadata.get("truncated") else "malformed"
            return _failed(example, turn, board, legal_tiles, raw_response, None, outcome, metadata, steps)
        if tile not in legal_tiles:
            return _failed(example, turn, board, legal_tiles, raw_response, tile, "illegal", metadata, steps)
        next_board = slide_tile(board, tile)
        progress = distance_progress_reward(board, next_board)
        solved = is_solved(next_board)
        terminal = solved_reward(example.optimal_length, turn) if solved else 0.0
        steps.append(StepResult(turn, board, legal_tiles, raw_response, tile, next_board, "solved" if solved else "valid", metadata, progress + terminal, progress, terminal))
        history.append(HistoryTurn(
            board=board,
            tile=tile,
            reasoning=metadata.get("reasoning_content", "") if metadata else "",
            reasoning_details=metadata.get("reasoning_details") if metadata else None,
        ))
        board = next_board
        if solved:
            return EpisodeResult(example, "solved", sum(step.reward for step in steps), turn, board, steps)

    steps[-1].reward += TIMEOUT_REWARD
    steps[-1].terminal_reward += TIMEOUT_REWARD
    return EpisodeResult(example, "timeout", sum(step.reward for step in steps), max_turns, board, steps)
