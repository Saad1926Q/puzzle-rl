"""Async TRL rollout worker with an exact bounded board-history policy."""

from __future__ import annotations

import json
import time
from typing import Any

from trl.experimental.async_grpo.async_rollout_worker import (
    AsyncRolloutWorker,
    Messages,
    TurnRecord,
    _AsyncRolloutLoop,
    _chain_to_sequences,
)
from trl.chat_template_utils import parse_response

from puzzle3.history import LastTurns
from .environment import PuzzleEnvironment
from .rollout import PuzzleRolloutEngine


class _PuzzleAsyncRolloutLoop(_AsyncRolloutLoop):
    """TRL's async loop with puzzle-specific context reconstruction."""

    def __init__(
        self,
        *,
        history_turns: int = 4,
        include_reasoning: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._history_policy = LastTurns(history_turns)
        self._include_reasoning = include_reasoning

    async def _generate_one(
        self,
        prompt: Messages,
        tool_dict: dict[str, Any],
        tools: list[Any],
        group_id: int = 0,
    ) -> tuple[list[dict[str, Any]], list[int], list[Any], int, int, float | None]:
        """Generate one puzzle episode while rebuilding the visible context each turn."""

        slide_tool = tool_dict.get("slide_tile")
        environment = getattr(slide_tool, "__self__", None)
        if not isinstance(environment, PuzzleEnvironment):
            return await super()._generate_one(prompt, tool_dict, tools, group_id)

        episode = environment.episode
        if episode is None:
            raise RuntimeError("PuzzleEnvironment must be reset before generation")

        started_at = time.monotonic()
        rollout_id = __import__("uuid").uuid4().hex
        context = PuzzleRolloutEngine(
            initial_prompt=prompt,
            history_policy=self._history_policy,
            include_reasoning=self._include_reasoning,
        )
        turns: list[TurnRecord] = []
        completion: list[dict[str, Any]] = []
        completion_ids: list[int] = []
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0
        loop_exhausted = False

        while not episode.done:
            messages = context.messages(episode.board)
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
                episode.fail("malformed")
                break
            if self.max_tool_calling_iterations is not None and iteration_num >= self.max_tool_calling_iterations:
                episode.fail("truncated")
                loop_exhausted = True
                break

            tool_call_count += 1
            tool_call = tool_calls[0]
            if not isinstance(tool_call, dict):
                tool_failure_count += 1
                episode.fail("malformed")
                break
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                tool_failure_count += 1
                episode.fail("malformed")
                break
            name = function.get("name")
            self._counters[f"tools/{name}_call_total"] += 1
            self._rates["tools/parallel_calls_mean"][0] += 1
            self._rates["tools/parallel_calls_mean"][1] += 1
            if name != "slide_tile":
                tool_failure_count += 1
                self._counters["tools/unknown_name_total"] += 1
                episode.fail("malformed")
                result: Any = {"error": f"unknown tool {name}"}
            else:
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = None
                tile = arguments.get("tile") if isinstance(arguments, dict) else None
                if type(tile) is not int:
                    tool_failure_count += 1
                    episode.fail("malformed")
                    result = {"error": "slide_tile requires an integer tile"}
                else:
                    previous_board = episode.board
                    started_tool = time.monotonic()
                    try:
                        result = slide_tool(tile=tile)
                    except Exception as error:
                        tool_failure_count += 1
                        result = {"error": str(error)}
                    self._rates["tools/latency_s"][0] += time.monotonic() - started_tool
                    self._rates["tools/latency_s"][1] += 1
                    transition = environment.last_transition
                    if transition is not None and transition.status in {"valid", "solved", "timeout"}:
                        context.record_action(
                            board=previous_board,
                            tile=tile,
                            assistant_message=assistant_message,
                        )

            completion.append({"role": "tool", "name": "slide_tile", "content": str(result)})
            if episode.done:
                break
            iteration_num += 1

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
            loop_exhausted=loop_exhausted,
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


def build_puzzle_worker_kwargs(
    *,
    model_name: str,
    dataset: Any,
    processing_class: Any,
    num_generations: int,
    max_inflight_tasks: int,
    vllm_server_url: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float | None,
    repetition_penalty: float,
    request_timeout: int,
    chat_template_kwargs: dict[str, Any] | None,
    max_tool_calling_iterations: int | None,
    log_completions: bool,
    num_completions_to_print: int | None,
    fork_threshold_tokens: int,
    history_turns: int = 4,
    queue_maxsize: int = 0,
) -> dict[str, Any]:
    """Build explicit, picklable kwargs for ``PuzzleAsyncRolloutWorker``."""

    return {
        "model_name": model_name,
        "dataset": dataset,
        "reward_funcs": [],
        "processing_class": processing_class,
        "tools": [],
        "environment_factory": PuzzleEnvironment,
        "num_generations": num_generations,
        "max_inflight_tasks": max_inflight_tasks,
        "queue_maxsize": queue_maxsize,
        "vllm_server_url": vllm_server_url,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "request_timeout": request_timeout,
        "chat_template_kwargs": chat_template_kwargs,
        "max_tool_calling_iterations": max_tool_calling_iterations,
        "log_completions": log_completions,
        "num_completions_to_print": num_completions_to_print,
        "fork_threshold_tokens": fork_threshold_tokens,
        "history_turns": history_turns,
    }
