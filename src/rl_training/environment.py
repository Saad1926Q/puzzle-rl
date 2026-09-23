"""TRL environment adapter backed by the shared puzzle episode state machine."""

from __future__ import annotations

from typing import Any, Sequence

from puzzle3.board import adjacent_tiles
from puzzle3.episode import DEFAULT_MAX_TURNS, PuzzleEpisode, Transition


class PuzzleEnvironment:
    """Stateful environment exposing one ``slide_tile`` tool to TRL."""

    def __init__(self) -> None:
        self.episode: PuzzleEpisode | None = None
        self.last_transition: Transition | None = None

    def reset(
        self,
        board: Sequence[int],
        optimal_length: int,
        max_turns: int = DEFAULT_MAX_TURNS,
        **_: Any,
    ) -> str:
        """Start a puzzle episode and return its initial board observation."""

        self.episode = PuzzleEpisode(
            initial_board=tuple(board),
            optimal_length=optimal_length,
            max_turns=max_turns,
        )
        self.last_transition = None
        return self._observation()

    def slide_tile(self, tile: int) -> str:
        """Slide one numbered tile adjacent to the blank into the blank.

        Args:
            tile: The adjacent numbered tile to slide into the blank.

        Returns:
            The resulting board, legal actions, and episode status.
        """

        episode = self._require_episode()
        transition = episode.step(tile)
        self.last_transition = transition
        if transition.status == "illegal":
            raise ValueError(
                f"illegal tile {tile}; legal tiles are {list(transition.legal_tiles)}"
            )
        return self._observation()

    def get_reward(self) -> float:
        """Return the environment-authoritative final episode reward."""

        return self._require_episode().reward

    @property
    def reward(self) -> float:
        """Expose the current episode reward for custom reward functions."""

        return self.get_reward()

    @property
    def done(self) -> bool:
        """Whether the current episode has terminated."""

        return self._require_episode().done

    @property
    def outcome(self) -> str:
        """Return the current episode outcome."""

        return self._require_episode().outcome

    def _require_episode(self) -> PuzzleEpisode:
        if self.episode is None:
            raise RuntimeError("reset must be called before interacting with the environment")
        return self.episode

    def _observation(self) -> str:
        episode = self._require_episode()
        board = episode.board
        rows = [board[row * 3 : (row + 1) * 3] for row in range(3)]
        rendered = "\n".join(" ".join(str(tile) for tile in row) for row in rows)
        legal = list(adjacent_tiles(board)) if not episode.done else []
        return (
            "Current board (0 is the blank):\n"
            f"{rendered}\n"
            f"Legal tiles: {legal}\n"
            f"Status: {episode.outcome}"
        )
