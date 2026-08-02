"""Puzzle episode state."""

from __future__ import annotations

from typing import Any

from env.protocol import build_board_prompt, parse_move
from env.rewards import final_reward, metrics
from puzzle.board import apply_move, is_solved, legal_moves

MAX_TURNS = 120
TERMINAL_REASONS = ("solved", "illegal_move", "format_failure", "max_turns")
ROW_FIELDS = ("initial_board", "optimal_length", "bucket", "optimal_moves", "scramble_depth")


class FifteenPuzzleSession:
    """One 15-puzzle episode."""

    def __init__(
        self,
        initial_board: tuple[int, ...] | list[int],
        optimal_length: int,
        bucket: str,
        optimal_moves: tuple[str, ...] | list[str] | None = None,
        scramble_depth: int | None = None,
        max_turns: int = MAX_TURNS,
    ) -> None:
        self.initial_board = tuple(initial_board)
        self.optimal_length = int(optimal_length)
        self.bucket = bucket
        self.optimal_moves = tuple(optimal_moves) if optimal_moves is not None else None
        self.scramble_depth = scramble_depth
        self.max_turns = max_turns
        self.reset()

    @classmethod
    def from_row(cls, row: dict[str, Any], max_turns: int = MAX_TURNS) -> "FifteenPuzzleSession":
        """Build from a dataset row."""

        board = row.get("initial_board", row.get("board"))
        return cls(
            initial_board=board,
            optimal_length=row["optimal_length"],
            bucket=row["bucket"],
            optimal_moves=row.get("optimal_moves"),
            scramble_depth=row.get("scramble_depth"),
            max_turns=max_turns,
        )

    @classmethod
    def from_prompt(cls, prompt: dict[str, Any], max_turns: int = MAX_TURNS) -> "FifteenPuzzleSession":
        """Build from a TRL prompt payload."""

        return cls(
            initial_board=prompt["initial_board"],
            optimal_length=prompt["optimal_length"],
            bucket=prompt["bucket"],
            optimal_moves=prompt.get("optimal_moves"),
            scramble_depth=prompt.get("scramble_depth"),
            max_turns=max_turns,
        )

    def reset(self) -> None:
        """Start over from the initial board."""

        self.current_board = tuple(self.initial_board)
        self.moves_taken: list[str] = []
        self.solved = False
        self.done = False
        self.illegal_move = False
        self.format_failure = False
        self.terminal_reason: str | None = None

    def apply_model_response(self, response_text: str) -> str | None:
        """Apply one model turn and return the next board prompt."""

        if self.done:
            raise RuntimeError(f"session already terminal: {self.terminal_reason}")

        move = parse_move(response_text)
        if move is None:
            self.format_failure = True
            self.done = True
            self.terminal_reason = "format_failure"
            return None

        if move not in legal_moves(self.current_board):
            self.illegal_move = True
            self.done = True
            self.terminal_reason = "illegal_move"
            return None

        self.current_board = apply_move(self.current_board, move)
        self.moves_taken.append(move)

        if is_solved(self.current_board):
            self.solved = True
            self.done = True
            self.terminal_reason = "solved"
            return None

        if len(self.moves_taken) >= self.max_turns:
            self.done = True
            self.terminal_reason = "max_turns"
            return None

        return build_board_prompt(self.current_board)

    def final_reward(self) -> float:
        """Score the episode."""

        return final_reward(
            solved=self.solved,
            moves_taken=len(self.moves_taken),
            optimal_length=self.optimal_length,
            illegal_move=self.illegal_move,
            format_failure=self.format_failure,
        )

    def metrics(self) -> dict[str, Any]:
        """Return metrics plus episode context."""

        out = metrics(
            solved=self.solved,
            illegal_move=self.illegal_move,
            format_failure=self.format_failure,
            moves_taken=len(self.moves_taken),
            optimal_length=self.optimal_length,
        )
        out.update(
            {
                "turns": float(len(self.moves_taken)),
                "bucket": self.bucket,
                "terminal_reason": self.terminal_reason,
                "reward": self.final_reward(),
            }
        )
        return out

    @property
    def moves_taken_count(self) -> int:
        return len(self.moves_taken)

    def current_board_prompt(self) -> str:
        return build_board_prompt(self.current_board)
