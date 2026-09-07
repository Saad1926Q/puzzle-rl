"""Typed internal representation of replay-verified SFT trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from puzzle3.board import Board, GOAL, TileAction, adjacent_tiles, is_solved, slide_tile


@dataclass(frozen=True)
class TrajectoryStep:
    turn: int
    board: Board
    legal_tiles: tuple[TileAction, ...]
    tile: TileAction
    next_board: Board
    status: str
    raw_response: str | None
    reasoning: str
    response_metadata: dict[str, Any] | None

    @classmethod
    def from_dict(cls, value: Any, *, expected_turn: int, board: Board) -> TrajectoryStep:
        if not isinstance(value, dict) or value.get("turn") != expected_turn:
            raise ValueError("trajectory turns must be contiguous")
        if _board(value.get("board"), "step board") != board:
            raise ValueError(f"turn {expected_turn} has an incorrect board")
        tile = value.get("tile")
        if type(tile) is not int or tile not in adjacent_tiles(board):
            raise ValueError(f"turn {expected_turn} has an illegal tile")
        next_board = slide_tile(board, tile)
        if _board(value.get("next_board"), "next_board") != next_board:
            raise ValueError(f"turn {expected_turn} has an incorrect next board")
        legal_tiles = value.get("legal_tiles")
        if not isinstance(legal_tiles, list) or tuple(legal_tiles) != adjacent_tiles(board):
            raise ValueError(f"turn {expected_turn} has incorrect legal tiles")
        return cls(
            turn=expected_turn,
            board=board,
            legal_tiles=tuple(legal_tiles),
            tile=tile,
            next_board=next_board,
            status=str(value.get("status", "")),
            raw_response=value.get("raw_response"),
            reasoning=str(value.get("reasoning", "")),
            response_metadata=value.get("response_metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "board": list(self.board),
            "legal_tiles": list(self.legal_tiles),
            "tile": self.tile,
            "next_board": list(self.next_board),
            "status": self.status,
            "raw_response": self.raw_response,
            "reasoning": self.reasoning,
            "response_metadata": self.response_metadata,
        }


@dataclass(frozen=True)
class Trajectory:
    source_id: str
    initial_board: Board
    optimal_length: int
    rollout_id: int
    moves_taken: int
    final_board: Board
    teacher_model: str | None
    steps: tuple[TrajectoryStep, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Trajectory:
        initial_board = _board(value.get("initial_board"), "initial_board")
        raw_steps = value.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("trajectory must contain at least one step")
        board = initial_board
        steps: list[TrajectoryStep] = []
        for turn, raw_step in enumerate(raw_steps, start=1):
            step = TrajectoryStep.from_dict(raw_step, expected_turn=turn, board=board)
            steps.append(step)
            board = step.next_board
        if not is_solved(board) or board != GOAL:
            raise ValueError("trajectory does not end at the goal")
        if value.get("moves_taken") != len(steps):
            raise ValueError("moves_taken does not match the step count")
        if _board(value.get("final_board"), "final_board") != board:
            raise ValueError("final_board does not match the replay")
        return cls(
            source_id=str(value.get("source_id")),
            initial_board=initial_board,
            optimal_length=int(value.get("optimal_length")),
            rollout_id=int(value.get("rollout_id")),
            moves_taken=len(steps),
            final_board=board,
            teacher_model=value.get("teacher_model"),
            steps=tuple(steps),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "initial_board": list(self.initial_board),
            "optimal_length": self.optimal_length,
            "rollout_id": self.rollout_id,
            "moves_taken": self.moves_taken,
            "final_board": list(self.final_board),
            "teacher_model": self.teacher_model,
            "steps": [step.to_dict() for step in self.steps],
        }


def _board(value: Any, field: str) -> Board:
    if not isinstance(value, list | tuple) or len(value) != 9:
        raise ValueError(f"{field} must contain nine integers")
    board = tuple(value)
    if any(type(tile) is not int for tile in board) or set(board) != set(GOAL):
        raise ValueError(f"{field} must be a permutation of 0 through 8")
    return board
