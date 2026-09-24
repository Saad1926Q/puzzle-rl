"""Async TRL rollout worker with an exact bounded board-history policy."""

from __future__ import annotations

import time
import uuid
from typing import Any

from trl.chat_template_utils import parse_response
from trl.experimental.async_grpo.async_rollout_worker import (
    AsyncRolloutWorker,
    Messages,
    TurnRecord,
    _AsyncRolloutLoop,
    _chain_to_sequences,
)

from evaluation.protocol import HistoryTurn, build_chat_completion_messages
from puzzle3.environment import DEFAULT_HISTORY_TURNS, PuzzleEnv


class _PuzzleAsyncRolloutLoop(_AsyncRolloutLoop):
    """TRL's async loop with bounded puzzle history."""

    def __init__(
        self,
        *,
        history_turns: int = DEFAULT_HISTORY_TURNS,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if history_turns < 0:
            raise ValueError("history_turns must be non-negative")
        self.history_turns = history_turns

    async def _generate_one(
        self,
        _prompt: Messages,
        tool_dict: dict[str, Any],
        tools: list[Any],
        group_id: int = 0,
    ) -> tuple[list[dict[str, Any]], list[int], list[Any], int, int, float | None]:
        """Generate one puzzle episode with a rolling board history."""

        slide_tool = tool_dict.get("slide_tile")
        environment = getattr(slide_tool, "__self__", None)
        if not isinstance(environment, PuzzleEnv):
            return await super()._generate_one(_prompt, tool_dict, tools, group_id)

        if environment.board is None:
            raise RuntimeError("PuzzleEnv must be reset before generation")

        started_at = time.monotonic()
        rollout_id = uuid.uuid4().hex
        history: list[HistoryTurn] = []
        turns: list[TurnRecord] = []
        completion: list[dict[str, Any]] = []
        completion_ids: list[int] = []
        tool_call_count = 0
        tool_failure_count = 0

        while not environment.done:
            visible_history = (
                tuple(history[-self.history_turns :]) if self.history_turns else ()
            )
            messages = build_chat_completion_messages(
                environment.board,
                visible_history,
                include_reasoning=True,
            )
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                return_dict=False,
                add_generation_prompt=True,
                tools=tools or None,
                chat_template=self.chat_template,
                **self.chat_template_kwargs,
            )
            turn_ids, turn_logprobs = await self._generate_one_turn(prompt_ids)
            assistant_message = parse_response(
                self.tokenizer,
                turn_ids,
                prefix=prompt_ids,
            )
            completion.append(assistant_message)
            completion_ids.extend(turn_ids)
            turns.append(TurnRecord(prompt_ids, turn_ids, turn_logprobs))

            tool_calls = assistant_message.get("tool_calls")
            if not isinstance(tool_calls, list) or len(tool_calls) != 1:
                environment._fail("malformed")
                break

            tool_call = tool_calls[0]
            if not isinstance(tool_call, dict) or not isinstance(
                tool_call.get("function"), dict
            ):
                tool_failure_count += 1
                environment._fail("malformed")
                break
            function = tool_call["function"]
            tile = (
                function.get("arguments", {}).get("tile")
                if isinstance(function.get("arguments"), dict)
                else None
            )

            tool_messages, n_calls, n_failures = await self._execute_tool_calls(
                tool_calls, tool_dict
            )
            tool_call_count += n_calls
            tool_failure_count += n_failures
            completion.extend(tool_messages)
            if n_failures and not environment.done:
                environment._fail("malformed")
            move = environment.last_move
            if (
                move is not None
                and move.status in {"valid", "solved", "timeout"}
                and type(tile) is int
            ):
                history.append(
                    HistoryTurn(
                        board=move.board,
                        tile=tile,
                        reasoning=assistant_message.get("content") or "",
                        reasoning_details=assistant_message.get("reasoning_details"),
                    )
                )
            if environment.done:
                break

        sequences, tally = _chain_to_sequences(
            turns,
            rollout_id,
            self._fork_threshold_tokens,
        )
        self._push_rollout_metrics(
            turns=len(turns),
            sequences=len(sequences),
            completion_ids=completion_ids,
            tally=tally,
            loop_exhausted=False,
            duration_s=time.monotonic() - started_at,
        )
        return (
            completion,
            completion_ids,
            sequences,
            tool_call_count,
            tool_failure_count,
            None,
        )


class PuzzleAsyncRolloutWorker(AsyncRolloutWorker):
    """Default TRL async worker with puzzle context reconstruction.

    The parent worker retains grouping, queueing, scoring, staleness, health,
    metrics, and process lifecycle. Only the private generation loop is replaced,
    so this class is pinned to the TRL version declared by the project.
    """

    _loop_cls = _PuzzleAsyncRolloutLoop
