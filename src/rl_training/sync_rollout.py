"""Synchronous bounded-history puzzle rollouts for GRPO training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import nan
from typing import Any

from evaluation.protocol import HistoryTurn, build_chat_completion_messages
from puzzle3.environment import PuzzleEnv
from sft_generation.training import normalize_tool_arguments
from trl.chat_template_utils import parse_response


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One generation call and the exact context used to produce it."""

    prompt_ids: list[int]
    completion_ids: list[int]
    logprobs: list[float]


@dataclass(slots=True)
class EpisodeRollout:
    """One complete puzzle episode and its turn-level training records."""

    environment: PuzzleEnv
    turns: list[TurnRecord] = field(default_factory=list)
    history: list[HistoryTurn] = field(default_factory=list)
    truncated: bool = False
    tool_calls: int = 0
    tool_failures: int = 0

    @property
    def reward(self) -> float:
        return self.environment.get_reward()

    @property
    def outcome(self) -> str:
        return self.environment.outcome


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def _sampled_logprobs(values: Any) -> list[float]:
    """Extract vLLM's sampled-token logprob from its top-k response."""

    result: list[float] = []
    for value in values:
        if not value or value[0] is None:
            result.append(nan)
        else:
            result.append(float(value[0]))
    return result


def _looks_truncated(tokenizer: Any, completion_ids: list[int]) -> bool:
    if not completion_ids:
        return False
    terminal_ids = {
        token_id
        for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if token_id is not None
    }
    return completion_ids[-1] not in terminal_ids


def _tool_tile(message: Any) -> int | None:
    if not isinstance(message, dict):
        return None
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        return None
    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict) or function.get("name") != "slide_tile":
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    tile = arguments.get("tile")
    return tile if type(tile) is int else None


def _tool_schema(trainer: Any) -> list[Any]:
    """Return the environment tool schema already validated by GRPOTrainer."""

    environment_tools = getattr(trainer, "_env_tools", {})
    tools = environment_tools.get(None)
    if tools is None:
        raise RuntimeError("PuzzleSyncRollout requires GRPOTrainer(environment_factory=PuzzleEnv)")
    return tools


def _render_prompt(trainer: Any, environment: PuzzleEnv, history: list[HistoryTurn]) -> list[int]:
    messages = build_chat_completion_messages(
        environment.board,  # type: ignore[arg-type]
        tuple(history[-trainer.history_turns :]) if trainer.history_turns else (),
        include_reasoning=True,
    )
    messages = normalize_tool_arguments({"prompt": messages})["prompt"]
    tokenized = trainer.processing_class.apply_chat_template(
        messages,
        return_dict=False,
        add_generation_prompt=True,
        tools=_tool_schema(trainer),
        chat_template=trainer.chat_template,
        **trainer.chat_template_kwargs,
    )
    return _token_ids(tokenized)


def _sync_vllm_weights(trainer: Any) -> None:
    if not trainer.use_vllm:
        raise RuntimeError("PuzzleSyncRollout requires vLLM generation")
    if trainer.state.global_step != trainer._last_loaded_step:
        trainer.vllm_generation.sync_weights()
        trainer._last_loaded_step = trainer.state.global_step


def _generate_turns(trainer: Any, episodes: list[EpisodeRollout]) -> None:
    """Generate one turn for every still-running episode in a batched vLLM call."""

    active = [episode for episode in episodes if not episode.environment.done]
    if not active:
        return

    prompt_ids = [
        _render_prompt(trainer, episode.environment, episode.history) for episode in active
    ]
    _, completion_ids, logprobs, _ = trainer.vllm_generation.generate(
        prompts=prompt_ids,
        images=None,
        num_generations=1,
    )

    for episode, prompt, completion, turn_logprobs in zip(
        active, prompt_ids, completion_ids, logprobs, strict=True
    ):
        completion_ids_for_turn = _token_ids(completion)
        episode.turns.append(
            TurnRecord(
                prompt_ids=prompt,
                completion_ids=completion_ids_for_turn,
                logprobs=_sampled_logprobs(turn_logprobs),
            )
        )
        try:
            assistant_message = parse_response(
                trainer.processing_class,
                completion_ids_for_turn,
                prefix=prompt,
            )
            assistant_message = normalize_tool_arguments(
                {"completion": [assistant_message]}
            )["completion"][0]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            assistant_message = {}

        tile = _tool_tile(assistant_message)
        if tile is None:
            episode.truncated = _looks_truncated(
                trainer.processing_class, completion_ids_for_turn
            )
            episode.environment._fail(
                "truncated" if episode.truncated else "malformed"
            )
            episode.tool_failures += 1
            continue

        episode.tool_calls += 1
        move = episode.environment._move(tile)
        if move.status == "illegal":
            episode.tool_failures += 1
            continue

        if move.status in {"valid", "solved", "timeout"}:
            episode.history.append(
                HistoryTurn(
                    board=move.board,
                    tile=tile,
                    reasoning=assistant_message.get("content") or "",
                    reasoning_details=assistant_message.get("reasoning_details"),
                )
            )


def generate_episode_group(
    trainer: Any,
    inputs: list[dict[str, Any]],
) -> list[EpisodeRollout]:
    """Generate one synchronous group, preserving rolling history per episode."""

    if not inputs:
        raise ValueError("cannot generate an empty puzzle group")
    _sync_vllm_weights(trainer)
    episodes: list[EpisodeRollout] = []
    for row in inputs:
        environment = PuzzleEnv()
        environment.reset(
            board=row["board"],
            optimal_length=int(row["optimal_length"]),
            max_turns=int(row["max_turns"]),
        )
        episodes.append(EpisodeRollout(environment=environment))

    max_turns = max(episode.environment.max_turns for episode in episodes)
    for _ in range(max_turns):
        _generate_turns(trainer, episodes)
        if all(episode.environment.done for episode in episodes):
            break
    return episodes
