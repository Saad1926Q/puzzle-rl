"""Authoritative state machine for one 3x3 sliding-puzzle episode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from puzzle3.board import Board, TileAction, adjacent_tiles, is_solved, slide_tile
from puzzle3.rewards import distance_progress_reward, solved_reward
from puzzle3.solver import exact_distance

MAX_TURNS = 45
DEFAULT_MAX_TURNS = MAX_TURNS
ILLEGAL_OR_MALFORMED_REWARD = -1.0

EpisodeOutcome = Literal[
    "running",
    "solved",
    "illegal",
    "malformed",
    "truncated",
    "timeout",
    "api_error",
]


@dataclass(frozen=True, slots=True)
class Transition:
    """One attempted action and its environment-authoritative result."""

    board: Board
    next_board: Board | None
    tile: TileAction | None
    legal_tiles: tuple[TileAction, ...]
    status: EpisodeOutcome
    progress_reward: float
    terminal_reward: float
    reward: float
    done: bool


@dataclass(slots=True)
class PuzzleEpisode:
    """Run one validated puzzle with the shared evaluation reward semantics."""

    initial_board: Board
    optimal_length: int
    max_turns: int = DEFAULT_MAX_TURNS
    board: Board = field(init=False)
    moves_taken: int = field(init=False, default=0)
    outcome: EpisodeOutcome = field(init=False, default="running")
    done: bool = field(init=False, default=False)
    reward: float = field(init=False, default=0.0)
    progress_reward: float = field(init=False, default=0.0)
    terminal_reward: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.initial_board = tuple(self.initial_board)
        self._validate_inputs()
        self.reset()

    def _validate_inputs(self) -> None:
        if len(self.initial_board) != 9 or set(self.initial_board) != set(range(9)):
            raise ValueError("initial_board must be a permutation of 0 through 8")
        if type(self.optimal_length) is not int or self.optimal_length < 0:
            raise ValueError("optimal_length must be a non-negative integer")
        if exact_distance(self.initial_board) != self.optimal_length:
            raise ValueError("optimal_length must equal the board's exact solution distance")
        if type(self.max_turns) is not int or not 1 <= self.max_turns <= MAX_TURNS:
            raise ValueError(f"max_turns must be between 1 and {MAX_TURNS}")

    def reset(self) -> Board:
        """Reset the episode and return its initial board."""

        self.board = self.initial_board
        self.moves_taken = 0
        self.outcome = "solved" if is_solved(self.board) else "running"
        self.done = is_solved(self.board)
        self.progress_reward = 0.0
        self.terminal_reward = solved_reward(self.optimal_length, 0) if self.done else 0.0
        self.reward = self.terminal_reward
        return self.board

    def step(self, tile: TileAction) -> Transition:
        """Apply one tile action and return the resulting transition."""

        if self.done:
            raise RuntimeError("episode has already ended")

        board = self.board
        legal_tiles = adjacent_tiles(board)
        if type(tile) is not int or tile not in legal_tiles:
            return self._finish_failure(tile, legal_tiles, "illegal")

        next_board = slide_tile(board, tile)
        progress_reward = distance_progress_reward(board, next_board)
        self.progress_reward += progress_reward
        self.moves_taken += 1
        self.board = next_board

        if is_solved(next_board):
            self.outcome = "solved"
            self.done = True
            self.terminal_reward = solved_reward(self.optimal_length, self.moves_taken)
            self.reward = self.terminal_reward
            return Transition(
                board=board,
                next_board=next_board,
                tile=tile,
                legal_tiles=legal_tiles,
                status="solved",
                progress_reward=progress_reward,
                terminal_reward=self.terminal_reward,
                reward=self.terminal_reward,
                done=True,
            )

        if self.moves_taken >= self.max_turns:
            self.outcome = "timeout"
            self.done = True
            self.reward = self.progress_reward
            return Transition(
                board=board,
                next_board=next_board,
                tile=tile,
                legal_tiles=legal_tiles,
                status="timeout",
                progress_reward=progress_reward,
                terminal_reward=0.0,
                reward=progress_reward,
                done=True,
            )

        self.reward = self.progress_reward
        return Transition(
            board=board,
            next_board=next_board,
            tile=tile,
            legal_tiles=legal_tiles,
            status="valid",
            progress_reward=progress_reward,
            terminal_reward=0.0,
            reward=progress_reward,
            done=False,
        )

    def fail(
        self,
        outcome: Literal["malformed", "truncated", "api_error"],
    ) -> Transition:
        """Terminate without a board transition for a non-action failure."""

        if self.done:
            raise RuntimeError("episode has already ended")
        if outcome == "api_error":
            terminal_reward = 0.0
        else:
            terminal_reward = ILLEGAL_OR_MALFORMED_REWARD
        return self._finish_failure(None, adjacent_tiles(self.board), outcome, terminal_reward)

    def _finish_failure(
        self,
        tile: TileAction | None,
        legal_tiles: tuple[TileAction, ...],
        outcome: Literal["illegal", "malformed", "truncated", "api_error"],
        terminal_reward: float = ILLEGAL_OR_MALFORMED_REWARD,
    ) -> Transition:
        self.outcome = outcome
        self.done = True
        self.terminal_reward = terminal_reward
        self.reward = terminal_reward
        return Transition(
            board=self.board,
            next_board=None,
            tile=tile,
            legal_tiles=legal_tiles,
            status=outcome,
            progress_reward=0.0,
            terminal_reward=terminal_reward,
            reward=terminal_reward,
            done=True,
        )
