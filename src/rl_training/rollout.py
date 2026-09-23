"""Prompt and trace handling for bounded-history puzzle rollouts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from evaluation.protocol import HistoryTurn, build_chat_completion_messages
from puzzle3.board import Board
from puzzle3.history import LastTurns

Messages = list[dict[str, Any]]


@dataclass(slots=True)
class PuzzleRolloutEngine:
    """Build the exact prompt seen by each turn of one puzzle rollout.

    The initial prompt is retained for the first generation. After the first
    completed action, prompts are rebuilt from the current board and the
    bounded history policy. Generated token records remain owned by the TRL
    worker; this object only owns visible conversation state.
    """

    initial_prompt: Messages
    history_policy: LastTurns = field(default_factory=LastTurns)
    include_reasoning: bool = True
    history: list[HistoryTurn] = field(default_factory=list)

    def messages(self, board: Board) -> Messages:
        """Return a fresh message list for the next model generation."""

        if not self.history:
            return deepcopy(self.initial_prompt)
        return build_chat_completion_messages(
            board,
            self.history_policy.select(self.history),
            include_reasoning=self.include_reasoning,
        )

    def record_action(
        self,
        *,
        board: Board,
        tile: int,
        assistant_message: dict[str, Any],
    ) -> None:
        """Record one completed action for future bounded context."""

        self.history.append(
            HistoryTurn(
                board=board,
                tile=tile,
                reasoning=assistant_message.get("content") or "",
                reasoning_details=assistant_message.get("reasoning_details"),
            )
        )
