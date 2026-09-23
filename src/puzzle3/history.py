"""Reusable policies for selecting completed-turn context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

T = TypeVar("T")

DEFAULT_HISTORY_TURNS = 4


@dataclass(frozen=True, slots=True)
class LastTurns:
    """Select the most recent completed turns without mutating the source."""

    max_turns: int = DEFAULT_HISTORY_TURNS

    def __post_init__(self) -> None:
        if self.max_turns < 0:
            raise ValueError("max_turns must be non-negative")

    def select(self, history: Sequence[T]) -> tuple[T, ...]:
        """Return at most ``max_turns`` complete turns in chronological order."""

        if self.max_turns == 0:
            return ()
        return tuple(history[-self.max_turns :])
