"""OpenEnv rollout glue for GRPO."""

from __future__ import annotations

from typing import Any, Callable

from openenv.core.harness import (
    HarnessAdapter,
    HarnessRolloutResult,
    HarnessRunLimits,
    ModelStep,
    ModelStepResult,
    ResourceSession,
    ResourceSessionFactory,
    RolloutEvent,
    VerifyResult,
    build_harness_rollout_func,
)
from openenv.core.llm_client import LLMResponse
from trl.experimental.openenv import generate_rollout_completions

from env.protocol import build_system_prompt
from env.session import MAX_TURNS, FifteenPuzzleSession

DEFAULT_MAX_TURNS = MAX_TURNS


class FifteenPuzzleSessionAdapter(ResourceSession):
    """OpenEnv wrapper around one puzzle session."""

    def __init__(self, task: Any, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        if isinstance(task, dict) and "messages" in task:
            self._session = FifteenPuzzleSession.from_prompt(task, max_turns=max_turns)
        else:
            self._session = FifteenPuzzleSession.from_row(
                dict(task), max_turns=max_turns
            )

    def initial_messages(self) -> list[dict[str, Any]]:
        return [{"role": "user", "content": self._session.current_board_prompt()}]

    def list_tools(self) -> list:
        return []

    def call_tool(self, name: str, arguments: dict[str, Any]):
        raise NotImplementedError("the 15-puzzle harness uses <move> text, not tools")

    def step(self, response_text: str) -> str | None:
        """Apply one model response; return the next board prompt or None."""

        return self._session.apply_model_response(response_text)

    def verify(
        self,
        transcript: list[dict[str, Any]],
        final_state: Any | None = None,
    ) -> VerifyResult:
        return VerifyResult(
            env_reward=self._session.final_reward(),
            done=self._session.done,
            metrics=dict(self._session.metrics()),
        )

    def close(self) -> None:
        return None

    @property
    def session(self) -> FifteenPuzzleSession:
        return self._session


class FifteenPuzzleSessionFactory(ResourceSessionFactory):
    """Create one session per rollout."""

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self.max_turns = max_turns

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> FifteenPuzzleSessionAdapter:
        return FifteenPuzzleSessionAdapter(task, max_turns=self.max_turns)


class FifteenPuzzleHarnessAdapter(HarnessAdapter):
    """Harness loop for <think>/<move> text."""

    def run_white_box(
        self,
        model_step: ModelStep,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        run_limits = limits or HarnessRunLimits()

        adapter = session

        messages = list(adapter.initial_messages())

        result = HarnessRolloutResult(messages=list(messages))

        for turn_index in range(run_limits.max_turns):
            step_result = model_step(messages, [], dict(run_limits.sampling))

            messages.append(step_result.response.to_message_dict())

            result.prompt_ids.extend(step_result.prompt_ids)
            result.completion_ids.extend(step_result.completion_ids)
            result.logprobs.extend(step_result.logprobs)

            result.events.append(
                RolloutEvent(
                    type="model_response",
                    payload={
                        "turn": turn_index,
                        "content": step_result.response.content,
                    },
                )
            )

            observation = adapter.step(step_result.response.content)

            if adapter.session.done:
                result.done = True
                result.metrics["turns"] = turn_index + 1
                break

            if observation is not None:
                messages.append({"role": "user", "content": observation})

        result.messages = list(messages)
        result.metrics.setdefault(
            "turns", len([e for e in result.events if e.type == "model_response"])
        )
        return result

    def run_black_box(
        self,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        raise NotImplementedError("training uses the white-box rollout path")


def build_model_step(
    trainer: Any,
    tokenizer: Any,
    system_prompt: str | None = None,
) -> Callable[[list[dict[str, Any]], list, dict[str, Any]], ModelStepResult]:
    """Build the trainer-owned model step."""

    system_prompt = system_prompt or build_system_prompt()

    def _last_user_content(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, list):
                    parts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    content = "\n".join(parts)
                return str(content)
        raise ValueError("no user message in transcript")

    def model_step(
        messages: list[dict[str, Any]],
        tools: list,
        sampling: dict[str, Any],
    ) -> ModelStepResult:
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _last_user_content(messages)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )
        rollout_output = generate_rollout_completions(trainer, [prompt_text])[0]
        completion_text = rollout_output.get("text") or tokenizer.decode(
            rollout_output["completion_ids"],
            skip_special_tokens=True,
        )
        return ModelStepResult(
            response=LLMResponse(content=completion_text),
            prompt_ids=list(rollout_output["prompt_ids"]),
            completion_ids=list(rollout_output["completion_ids"]),
            logprobs=list(rollout_output["logprobs"]),
        )

    return model_step


def build_rollout_func(
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    system_prompt: str | None = None,
    session_factory: ResourceSessionFactory | None = None,
) -> Callable[[list[Any], Any], dict[str, list[Any]]]:
    """Build the TRL rollout function."""

    factory = session_factory or FifteenPuzzleSessionFactory(max_turns=max_turns)

    def _model_step_builder(trainer: Any, session: ResourceSession) -> ModelStep:
        return build_model_step(trainer, trainer.processing_class, system_prompt)

    base = build_harness_rollout_func(
        session_factory=factory,
        harness_adapter=FifteenPuzzleHarnessAdapter(),
        model_step_builder=_model_step_builder,
        limits=HarnessRunLimits(max_turns=max_turns),
        reward_key="env_reward",
    )

    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, list[Any]]:
        output = base(prompts, trainer)
        output["env_mask"] = [
            [1] * len(completion_ids) for completion_ids in output["completion_ids"]
        ]
        return output

    return rollout_func


def validate_rollout_trace(
    prompt_ids: list[int],
    completion_ids: list[int],
    logprobs: list[float],
    *,
    turn_prompt_ids: list[list[int]] | None = None,
    turn_completion_ids: list[list[int]] | None = None,
    turn_logprobs: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Check rollout token arrays."""

    if turn_prompt_ids is not None:
        flat_prompts = [t for turn in turn_prompt_ids for t in turn]
        if flat_prompts != list(prompt_ids):
            raise AssertionError(
                "per-turn prompt ids do not reconstruct the flat prompt ids"
            )
    if turn_completion_ids is not None:
        flat_completions = [t for turn in turn_completion_ids for t in turn]
        if flat_completions != list(completion_ids):
            raise AssertionError(
                "per-turn completion ids do not reconstruct the flat completion ids"
            )
    if turn_logprobs is not None:
        flat_logprobs = [lp for turn in turn_logprobs for lp in turn]
        if flat_logprobs != list(logprobs):
            raise AssertionError(
                "per-turn logprobs do not reconstruct the flat logprobs"
            )

    if len(logprobs) != len(completion_ids):
        raise AssertionError(
            f"missing logprobs: {len(logprobs)} logprobs for {len(completion_ids)} completion tokens"
        )

    n_turns = len(turn_prompt_ids or [])
    return {
        "turns": n_turns,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
        "logprobs": len(logprobs),
    }
