"""State and tool interface for one 3x3 sliding-puzzle episode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from puzzle3.board import Board, TileAction, adjacent_tiles, is_solved, slide_tile
from puzzle3.rewards import distance_progress_reward, solved_reward
from puzzle3.solver import exact_distance

MAX_TURNS = 45
DEFAULT_MAX_TURNS = MAX_TURNS
DEFAULT_HISTORY_TURNS = 4
ILLEGAL_OR_MALFORMED_REWARD = -1.0

MoveStatus = Literal[
    "valid",
    "solved",
    "timeout",
    "illegal",
    "malformed",
    "truncated",
    "api_error",
]


@dataclass(frozen=True, slots=True)
class MoveResult:
    """Details needed by evaluation after one attempted action."""

    board: Board
    next_board: Board | None
    tile: TileAction | None
    legal_tiles: tuple[TileAction, ...]
    status: MoveStatus
    progress_reward: float
    terminal_reward: float
    reward: float
    done: bool


class PuzzleEnv:
    """Stateful puzzle environment used by evaluation and TRL."""

    def __init__(self) -> None:
        self.board: Board | None = None
        self.initial_board: Board | None = None
        self.optimal_length = 0
        self.max_turns = DEFAULT_MAX_TURNS
        self.moves = 0
        self.progress_reward = 0.0
        self.reward = 0.0
        self.terminal_reward = 0.0
        self.outcome: str = "running"
        self.done = False
        self.last_move: MoveResult | None = None

    def reset(
        self,
        board: Sequence[int],
        optimal_length: int,
        max_turns: int = DEFAULT_MAX_TURNS,
        **_: Any,
    ) -> str:
        """Start a validated puzzle and return its first observation."""

        board = tuple(board)
        if len(board) != 9 or set(board) != set(range(9)):
            raise ValueError("board must be a permutation of 0 through 8")
        if type(optimal_length) is not int or optimal_length < 0:
            raise ValueError("optimal_length must be a non-negative integer")
        if exact_distance(board) != optimal_length:
            raise ValueError(
                "optimal_length must equal the board's exact solution distance"
            )
        if type(max_turns) is not int or not 1 <= max_turns <= MAX_TURNS:
            raise ValueError(f"max_turns must be between 1 and {MAX_TURNS}")

        self.board = board
        self.initial_board = board
        self.optimal_length = optimal_length
        self.max_turns = max_turns
        self.moves = 0
        self.progress_reward = 0.0
        self.terminal_reward = (
            solved_reward(optimal_length, 0) if is_solved(board) else 0.0
        )
        self.reward = self.terminal_reward
        self.outcome = "solved" if is_solved(board) else "running"
        self.done = is_solved(board)
        self.last_move = None
        return self._observation()

    def _move(self, tile: TileAction) -> MoveResult:
        board = self._require_board()
        if self.done:
            raise RuntimeError("episode has already ended")

        legal_tiles = adjacent_tiles(board)
        if type(tile) is not int or tile not in legal_tiles:
            return self._finish_failure(tile, legal_tiles, "illegal")

        next_board = slide_tile(board, tile)
        progress_reward = distance_progress_reward(board, next_board)
        self.board = next_board
        self.moves += 1
        self.progress_reward += progress_reward

        if is_solved(next_board):
            self.outcome = "solved"
            self.done = True
            self.terminal_reward = solved_reward(self.optimal_length, self.moves)
            self.reward = self.terminal_reward
            return self._remember(
                MoveResult(
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
            )

        if self.moves >= self.max_turns:
            self.outcome = "timeout"
            self.done = True
            self.reward = self.progress_reward
            return self._remember(
                MoveResult(
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
            )

        self.reward = self.progress_reward
        return self._remember(
            MoveResult(
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
        )

    def slide_tile(self, tile: int) -> str:
        """Slide one numbered tile adjacent to the blank into the blank.

        Args:
            tile: The adjacent numbered tile to slide into the blank.

        Returns:
            The resulting board, legal actions, and episode status.
        """

        result = self._move(tile)
        if result.status == "illegal":
            raise ValueError(
                f"illegal tile {tile}; legal tiles are {list(result.legal_tiles)}"
            )
        return self._observation()

    def _fail(
        self, status: Literal["malformed", "truncated", "api_error"]
    ) -> MoveResult:
        return self._finish_failure(
            None,
            tuple(adjacent_tiles(self._require_board())),
            status,
            terminal_reward=0.0
            if status == "api_error"
            else ILLEGAL_OR_MALFORMED_REWARD,
        )

    def get_reward(self) -> float:
        """Return the final environment reward for TRL."""

        self._require_board()
        return self.reward

    def _observation(self) -> str:
        board = self._require_board()
        rows = [board[row * 3 : (row + 1) * 3] for row in range(3)]
        rendered = "\n".join(" ".join(str(tile) for tile in row) for row in rows)
        legal = list(adjacent_tiles(board)) if not self.done else []
        return (
            "Current board (0 is the blank):\n"
            f"{rendered}\n"
            f"Legal tiles: {legal}\n"
            f"Status: {self.outcome}"
        )

    def _finish_failure(
        self,
        tile: TileAction | None,
        legal_tiles: tuple[TileAction, ...],
        status: Literal["illegal", "malformed", "truncated", "api_error"],
        *,
        terminal_reward: float = ILLEGAL_OR_MALFORMED_REWARD,
    ) -> MoveResult:
        self.outcome = status
        self.done = True
        self.terminal_reward = terminal_reward
        self.reward = self.progress_reward + terminal_reward
        return self._remember(
            MoveResult(
                board=self._require_board(),
                next_board=None,
                tile=tile,
                legal_tiles=legal_tiles,
                status=status,
                progress_reward=0.0,
                terminal_reward=terminal_reward,
                reward=terminal_reward,
                done=True,
            )
        )

    def _remember(self, result: MoveResult) -> MoveResult:
        self.last_move = result
        return result

    def _require_board(self) -> Board:
        if self.board is None:
            raise RuntimeError(
                "reset must be called before interacting with the environment"
            )
        return self.board
