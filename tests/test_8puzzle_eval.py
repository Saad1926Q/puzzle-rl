from __future__ import annotations

from dataclasses import dataclass, field
import json

import pytest
from datasets import Dataset

from evaluation.dataset import DatasetError, PuzzleExample, load_examples
from evaluation.evaluator import (
    evaluate,
    evaluate_episode,
    distance_progress_reward,
    solved_reward,
)
from evaluation.clients.crof import CrofAgent
from evaluation.clients.deepseek import DeepSeekAgent
from evaluation.clients.glm import GLMAgent
from evaluation.clients.openai_client import OpenAIAgent
from evaluation.clients.openrouter import OpenRouterAgent
from evaluation.clients.qwen import QwenAgent
from evaluation.protocol import (
    HistoryTurn,
    build_chat_completion_messages,
    build_openrouter_chat_completion_messages,
    build_openai_responses_input,
    get_api_key,
    parse_tile,
)
from puzzle3.board import GOAL
from puzzle3.solver import exact_distance


@dataclass
class SequenceAgent:
    responses: list[str]

    def next_action(self, board: tuple[int, ...]) -> str:
        return self.responses.pop(0)


def example(board: tuple[int, ...], optimal_length: int = 1) -> PuzzleExample:
    return PuzzleExample("test", board, tuple(), optimal_length, {})


def test_parser_requires_one_valid_tile() -> None:
    assert parse_tile('{"tile": 8}') == 8
    assert parse_tile('{"tile": 0}') is None
    assert parse_tile('{"tile": 9}') is None
    assert parse_tile('{"tile": true}') is None
    assert parse_tile('{"move": "left"}') is None


def test_messages_contain_current_board_but_no_history() -> None:
    messages = build_chat_completion_messages((1, 2, 3, 4, 5, 6, 7, 0, 8))
    assert len(messages) == 2
    assert "1 2 3" in messages[1]["content"]
    assert "7 0 8" in messages[1]["content"]
    assert "_" not in messages[1]["content"]
    assert "1 2 3 / 4 5 6 / 7 8 0" in messages[0]["content"]
    assert "history" not in messages[1]["content"].lower()



def test_messages_preserve_tool_protocol_and_optional_reasoning() -> None:
    history = (
        HistoryTurn((1, 2, 3, 4, 5, 6, 0, 7, 8), tile=7, reasoning="Move right."),
        HistoryTurn((1, 2, 3, 4, 5, 6, 7, 0, 8), tile=8, reasoning="Solve it."),
    )

    without_reasoning = build_chat_completion_messages(GOAL, history)
    with_reasoning = build_chat_completion_messages(GOAL, history, include_reasoning=True)

    assert [message["role"] for message in without_reasoning] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert "0 7 8" in without_reasoning[1]["content"]
    assert without_reasoning[2]["content"] == ""
    assert without_reasoning[2]["tool_calls"] == [
        {
            "id": "history_slide_0",
            "type": "function",
            "function": {"name": "slide_tile", "arguments": '{"tile": 7}'},
        }
    ]
    assert without_reasoning[3]["tool_call_id"] == "history_slide_0"
    assert "7 0 8" in without_reasoning[3]["content"]
    assert without_reasoning[4]["tool_calls"][0]["function"]["arguments"] == '{"tile": 8}'
    assert without_reasoning[5]["tool_call_id"] == "history_slide_1"
    assert "7 8 0" in without_reasoning[5]["content"]
    assert all(
        "Action: slide tile" not in str(message) for message in without_reasoning
    )
    assert with_reasoning[2]["content"] == "Move right."



def test_openrouter_messages_preserve_native_reasoning_fields() -> None:
    reasoning_details = [
        {"type": "reasoning.text", "text": "Native reasoning", "signature": "sig"}
    ]
    history = (
        HistoryTurn(
            (1, 2, 3, 4, 5, 6, 0, 7, 8),
            tile=7,
            reasoning="Plaintext fallback.",
            reasoning_details=reasoning_details,
        ),
        HistoryTurn(
            (1, 2, 3, 4, 5, 6, 7, 0, 8),
            tile=8,
            reasoning="Solve it.",
        ),
    )

    messages = build_openrouter_chat_completion_messages(
        GOAL, history, include_reasoning=True
    )

    assert messages[2]["content"] == ""
    assert messages[2]["reasoning_details"] is reasoning_details
    assert "reasoning" not in messages[2]
    assert messages[4]["content"] == ""
    assert messages[4]["reasoning"] == "Solve it."
    assert "reasoning_details" not in messages[4]


def test_openrouter_messages_omit_reasoning_when_not_requested() -> None:
    history = (
        HistoryTurn(
            (1, 2, 3, 4, 5, 6, 7, 0, 8),
            tile=8,
            reasoning="Solve it.",
            reasoning_details=[{"type": "reasoning.text", "text": "Solve it."}],
        ),
    )

    message = build_openrouter_chat_completion_messages(GOAL, history)[2]

    assert message["content"] == ""
    assert "reasoning" not in message
    assert "reasoning_details" not in message

def test_responses_input_preserves_function_call_history() -> None:
    history = (HistoryTurn((1, 2, 3, 4, 5, 6, 7, 0, 8), tile=8),)

    items = build_openai_responses_input(GOAL, history)

    assert items[2] == {
        "type": "function_call",
        "call_id": "history_slide_0",
        "name": "slide_tile",
        "arguments": '{"tile": 8}',
    }
    assert items[3]["type"] == "function_call_output"
    assert items[3]["call_id"] == "history_slide_0"
    assert "7 8 0" in items[3]["output"]


def test_responses_input_keeps_reasoning_outside_function_call() -> None:
    history = (
        HistoryTurn(
            (1, 2, 3, 4, 5, 6, 7, 0, 8),
            tile=8,
            reasoning="Move tile 8.",
        ),
    )

    items = build_openai_responses_input(GOAL, history, include_reasoning=True)

    assert items[2] == {"role": "assistant", "content": "Move tile 8."}
    assert items[3]["type"] == "function_call"
    assert "content" not in items[3]


def test_history_is_bounded_to_four_completed_turns() -> None:
    @dataclass
    class HistoryAgent:
        responses: list[str]
        received_history: list[tuple[HistoryTurn, ...]]

        def next_action(
            self,
            board: tuple[int, ...],
            history: tuple[HistoryTurn, ...],
            *,
            include_reasoning: bool,
        ) -> str:
            assert include_reasoning is False
            self.received_history.append(history)
            return self.responses.pop(0)
    agent = HistoryAgent(
        [
            '{"tile": 7}',
            '{"tile": 4}',
            '{"tile": 1}',
            '{"tile": 2}',
            '{"tile": 3}',
        ],
        [],
    )
    evaluate_episode(
        example((1, 2, 3, 4, 5, 6, 7, 0, 8)),
        agent,
        max_turns=5,
        keep_history=True,
    )

    assert [len(history) for history in agent.received_history] == [0, 1, 2, 3, 4]
    assert [turn.tile for turn in agent.received_history[-1]] == [7, 4, 1, 2]




def test_history_retains_reasoning_only_when_requested() -> None:
    @dataclass
    class ReasoningAgent:
        calls: int = 0
        received_history: list[tuple[HistoryTurn, ...]] = field(default_factory=list)

        def __post_init__(self) -> None:
            self.last_response_metadata: dict[str, object] = {}

        def next_action(
            self,
            board: tuple[int, ...],
            history: tuple[HistoryTurn, ...],
            *,
            include_reasoning: bool,
        ) -> str:
            self.received_history.append(history)
            self.last_response_metadata = {
                "reasoning_content": "Move toward goal.",
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "Move toward goal."}
                ],
            }
            self.calls += 1
            return '{"tile": 7}' if self.calls == 1 else '{"tile": 4}'

    agent = ReasoningAgent()
    evaluate_episode(
        example((1, 2, 3, 4, 5, 6, 7, 0, 8)),
        agent,
        max_turns=2,
        keep_history=True,
        keep_reasoning=True,
    )

    assert agent.received_history[1][0].reasoning == "Move toward goal."
    assert agent.received_history[1][0].reasoning_details == [
        {"type": "reasoning.text", "text": "Move toward goal."}
    ]


def test_reasoning_history_requires_history() -> None:
    with pytest.raises(ValueError, match="keep_reasoning requires keep_history"):
        evaluate_episode(
            example((1, 2, 3, 4, 5, 6, 7, 0, 8)),
            SequenceAgent(['{"tile": 8}']),
            keep_reasoning=True,
        )

def _fake_response(
    *,
    content="",
    tool_calls=None,
    finish_reason="tool_calls",
    reasoning=None,
    reasoning_content=None,
    reasoning_details=None,
):
    message = type(
        "Message",
        (),
        {
            "content": content,
            "tool_calls": tool_calls,
            "reasoning": reasoning,
            "reasoning_content": reasoning_content,
            "reasoning_details": reasoning_details,
        },
    )()
    choice = type("Choice", (), {"message": message, "finish_reason": finish_reason})()
    return type("Response", (), {"choices": [choice], "usage": None})()


def _fake_tool_call(arguments='{"tile":8}', name="slide_tile"):
    function = type("Function", (), {"name": name, "arguments": arguments})()
    return [type("ToolCall", (), {"function": function})()]


def _fake_responses_response(*, arguments='{"tile":8}', status="completed"):
    call = type(
        "FunctionCall",
        (),
        {"type": "function_call", "name": "slide_tile", "arguments": arguments},
    )()
    incomplete = type("Incomplete", (), {"reason": "max_output_tokens"})()
    return type(
        "Response",
        (),
        {
            "output": [call],
            "output_text": "",
            "status": status,
            "incomplete_details": incomplete if status == "incomplete" else None,
            "usage": None,
        },
    )()




def test_crof_uses_standard_chat_parameters_and_reasoning() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(
                tool_calls=_fake_tool_call('{"tile": 8}'),
                reasoning_content="Move tile 8 into the blank.",
            )

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = CrofAgent(
        api_key="not-used",
        client=client,
        model="glm-5.3-flash",
        reasoning_effort="medium",
    )

    assert parse_tile(agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))) == 8
    assert completions.kwargs["model"] == "glm-5.3-flash"
    assert completions.kwargs["tool_choice"] == "auto"
    assert completions.kwargs["reasoning_effort"] == "medium"
    assert "extra_body" not in completions.kwargs
    assert agent.last_response_metadata["reasoning_content"] == (
        "Move tile 8 into the blank."
    )


def test_crof_omits_reasoning_effort_when_thinking_is_disabled() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(tool_calls=_fake_tool_call())

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    CrofAgent(api_key="not-used", client=client, thinking=False).next_action(
        (1, 2, 3, 4, 5, 6, 7, 0, 8)
    )

    assert "reasoning_effort" not in completions.kwargs


def test_openrouter_sends_reproducible_routing_and_records_metadata() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            response = _fake_response(
                tool_calls=_fake_tool_call('{"tile": 8}'),
                reasoning_content="Slide tile 8.",
                reasoning_details=[
                    {"type": "reasoning.text", "text": "Slide tile 8."}
                ],
            )
            response.id = "gen-test"
            response.model = "qwen/qwen3.5-27b"
            response.model_extra = {
                "openrouter_metadata": {
                    "attempt": 1,
                    "attempts": [{"provider": "together", "status": 200}],
                }
            }
            return response

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = OpenRouterAgent(
        api_key="not-used",
        model="qwen/qwen3.5-27b",
        client=client,
        thinking=False,
        upstream_providers=("together",),
        quantizations=("bf16",),
        distillable_only=True,
        require_parameters=False,
    )

    assert parse_tile(agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))) == 8
    assert completions.kwargs["tool_choice"] == "auto"
    assert completions.kwargs["extra_body"] == {
        "reasoning": {"effort": "none", "exclude": False},
        "provider": {
            "allow_fallbacks": False,
            "require_parameters": False,
            "data_collection": "deny",
            "only": ["together"],
            "quantizations": ["bf16"],
            "enforce_distillable_text": True,
        },
    }
    assert agent.last_response_metadata["response_id"] == "gen-test"
    assert agent.last_response_metadata["resolved_model"] == "qwen/qwen3.5-27b"
    assert agent.last_response_metadata["openrouter_metadata"]["attempt"] == 1
    assert agent.last_response_metadata["reasoning_details"] == [
        {"type": "reasoning.text", "text": "Slide tile 8."}
    ]


def test_openrouter_uses_reasoning_effort_when_enabled() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(tool_calls=_fake_tool_call())

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    OpenRouterAgent(
        api_key="not-used",
        model="test/model",
        client=client,
        reasoning_effort="xhigh",
    ).next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))

    assert completions.kwargs["extra_body"]["reasoning"] == {
        "effort": "xhigh",
        "exclude": False,
    }


def test_openrouter_uses_provider_default_effort_when_unspecified() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(tool_calls=_fake_tool_call())

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    OpenRouterAgent(
        api_key="not-used",
        model="test/model",
        client=client,
    ).next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))

    assert completions.kwargs["extra_body"]["reasoning"] == {
        "enabled": True,
        "exclude": False,
    }


def test_openrouter_retries_an_upstream_provider_error() -> None:
    class ProviderError(RuntimeError):
        status_code = 400
        body = {"error": {"message": "Provider returned error"}}

    class Completions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("upstream rejected request")
            return _fake_response(tool_calls=_fake_tool_call())

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = OpenRouterAgent(
        api_key="not-used",
        model="test/model",
        client=client,
        retry_delay=0,
    )

    assert parse_tile(agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))) == 8
    assert completions.calls == 2

def test_deepseek_strict_tool_call_is_canonicalized() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(tool_calls=_fake_tool_call())

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert parse_tile(agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))) == 8
    assert completions.kwargs["tool_choice"]["function"]["name"] == "slide_tile"
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert agent.last_response_metadata["status"] == "tool_call"


def test_glm_tool_call_uses_zai_parameters() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(tool_calls=_fake_tool_call('{"tile":3}'))

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = GLMAgent(api_key="not-used", client=client, thinking=True)
    assert parse_tile(agent.next_action((1, 2, 0, 4, 5, 3, 7, 8, 6))) == 3
    assert completions.kwargs["tool_choice"] == "auto"
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "reasoning_effort" not in completions.kwargs
    assert completions.kwargs["max_tokens"] == 4096


def test_glm_disabled_thinking_is_explicit() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(tool_calls=_fake_tool_call())

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = GLMAgent(api_key="not-used", client=client, thinking=False)
    agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_qwen_uses_native_tool_call_and_sampling_parameters() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(
                tool_calls=_fake_tool_call('{"tile":8}'),
                reasoning="reason about the board",
            )

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = QwenAgent(
        client=client,
        thinking=True,
        reasoning_effort="xhigh",
        max_tokens=256,
        temperature=0.8,
        top_p=0.9,
        top_k=10,
        presence_penalty=1.5,
        repetition_penalty=1.1,
    )

    assert parse_tile(agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))) == 8
    assert completions.kwargs["tools"][0]["function"]["name"] == "slide_tile"
    assert completions.kwargs["tool_choice"] == "auto"
    assert completions.kwargs["parallel_tool_calls"] is False
    assert completions.kwargs["max_tokens"] == 256
    assert completions.kwargs["temperature"] == 0.8
    assert completions.kwargs["top_p"] == 0.9
    assert completions.kwargs["presence_penalty"] == 1.5
    assert completions.kwargs["reasoning_effort"] == "xhigh"
    assert completions.kwargs["extra_body"] == {
        "top_k": 10,
        "repetition_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert agent.last_response_metadata["status"] == "tool_call"
    assert agent.last_response_metadata["reasoning_content"] == "reason about the board"


def test_qwen_defaults_to_non_thinking_mode() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(tool_calls=_fake_tool_call())

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()

    QwenAgent(client=client).next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))

    assert completions.kwargs["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_qwen_cli_uses_local_defaults_and_needs_no_api_key() -> None:
    import runpy

    runner = runpy.run_path("scripts/run_eval_8puzzle.py")
    xhigh_args = runner["build_parser"]().parse_args(
        ["--provider", "qwen", "--reasoning-effort", "xhigh"]
    )
    history_args = runner["build_parser"]().parse_args(
        ["--provider", "qwen", "--keep-history", "--keep-reasoning"]
    )
    args = runner["build_parser"]().parse_args(["--provider", "qwen"])
    runner["resolve_provider_args"](args)

    assert args.model == "Qwen/Qwen3.5-0.8B"
    assert args.base_url == "http://localhost:8000/v1"
    assert xhigh_args.reasoning_effort == "xhigh"
    assert args.api_key_env is None
    assert args.thinking is False
    assert history_args.keep_history is True
    assert history_args.keep_reasoning is True
    crof_args = runner["build_parser"]().parse_args(["--provider", "crof"])
    runner["resolve_provider_args"](crof_args)
    assert crof_args.model == "glm-5.3-flash"
    assert crof_args.base_url == "https://crof.ai/v1"
    assert crof_args.api_key_env == "CROF_API_KEY"




def test_openrouter_cli_requires_model_and_uses_reproducible_defaults() -> None:
    import runpy

    runner = runpy.run_path("scripts/run_eval_8puzzle.py")
    parser = runner["build_parser"]()
    args = parser.parse_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "qwen/qwen3.5-27b",
            "--openrouter-upstream",
            "together",
            "--openrouter-quantization",
            "bf16",
            "--openrouter-distillable-only",
        ]
    )
    runner["resolve_provider_args"](args)

    assert args.base_url == "https://openrouter.ai/api/v1"
    assert args.api_key_env == "OPENROUTER_API_KEY"
    assert args.openrouter_allow_fallbacks is False
    assert args.openrouter_data_collection == "deny"
    assert args.openrouter_upstream == ["together"]
    assert args.openrouter_quantization == ["bf16"]
    assert args.openrouter_relax_parameters is False
    relaxed_args = parser.parse_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "qwen/qwen3.5-27b",
            "--openrouter-relax-parameters",
        ]
    )
    assert relaxed_args.openrouter_relax_parameters is True
    assert args.reasoning_effort is None

    missing_model = parser.parse_args(["--provider", "openrouter"])
    with pytest.raises(ValueError, match="--model is required"):
        runner["resolve_provider_args"](missing_model)



def test_openai_responses_tool_call_uses_responses_parameters() -> None:
    class Responses:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_responses_response(arguments='{"tile":6}')

    responses = Responses()
    client = type("Client", (), {"responses": responses})()
    agent = OpenAIAgent(api_key="not-used", client=client, thinking=True)
    history = (HistoryTurn((1, 2, 3, 4, 5, 6, 7, 0, 8), tile=8),)
    assert parse_tile(agent.next_action(GOAL, history)) == 6
    assert responses.kwargs["tool_choice"]["name"] == "slide_tile"
    assert responses.kwargs["parallel_tool_calls"] is False
    assert responses.kwargs["max_output_tokens"] == 4096
    assert responses.kwargs["reasoning"]["effort"] == "low"
    assert responses.kwargs["store"] is False
    assert responses.kwargs["input"][2]["type"] == "function_call"
    assert responses.kwargs["input"][3]["type"] == "function_call_output"


def test_openai_responses_truncated_tool_call_is_not_accepted() -> None:
    class Responses:
        def create(self, **kwargs):
            return _fake_responses_response(status="incomplete")

    client = type("Client", (), {"responses": Responses()})()
    agent = OpenAIAgent(api_key="not-used", client=client)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["truncated"] is True
    assert agent.last_response_metadata["status"] == "truncated"
    assert agent.last_response_metadata["incomplete_reason"] == "max_output_tokens"


def test_deepseek_thinking_mode_reads_tool_call_and_reasoning_metadata() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(
                tool_calls=_fake_tool_call('{"tile":3}'),
                reasoning_content="reason about the board",
            )

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=True)
    assert parse_tile(agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))) == 3
    assert completions.kwargs["tool_choice"] == "auto"
    assert agent.last_response_metadata["reasoning_content_length"] > 0


def test_deepseek_multiple_tool_calls_are_malformed() -> None:
    class Completions:
        def create(self, **kwargs):
            return _fake_response(
                tool_calls=_fake_tool_call('{"tile":8}') + _fake_tool_call('{"tile":3}')
            )

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["tool_call_count"] == 2
    assert agent.last_response_metadata["status"] == "malformed"


def test_deepseek_truncated_tool_call_is_not_accepted() -> None:
    class Completions:
        def create(self, **kwargs):
            return _fake_response(
                tool_calls=_fake_tool_call('{"tile":8}'), finish_reason="length"
            )

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["truncated"] is True
    assert agent.last_response_metadata["status"] == "truncated"


def test_deepseek_empty_or_truncated_response_is_diagnosed() -> None:
    class Completions:
        def create(self, **kwargs):
            return _fake_response(
                content="",
                tool_calls=None,
                finish_reason="length",
                reasoning_content="partial",
            )

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=True)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["status"] == "truncated"
    assert agent.last_response_metadata["truncated"] is True
    assert agent.last_response_metadata["reasoning_content_length"] == 7


def test_deepseek_malformed_tool_arguments_are_diagnosed() -> None:
    class Completions:
        def create(self, **kwargs):
            return _fake_response(tool_calls=_fake_tool_call('{"tile":0}'))

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["tool_name"] == "slide_tile"
    assert agent.last_response_metadata["status"] == "malformed"


def test_deepseek_provider_error_is_not_relabelled_as_malformed() -> None:
    class Completions:
        def create(self, **kwargs):
            raise RuntimeError("provider unavailable")

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))
    assert agent.last_response_metadata["status"] == "api_error"


def test_evaluation_records_api_error_without_aborting_other_rollouts() -> None:
    class APIErrorAgent:
        last_response_metadata: dict[str, object] = {}

        def next_action(self, board):
            self.last_response_metadata = {
                "status": "api_error",
                "error_type": "BadRequestError",
                "status_code": 400,
            }
            raise RuntimeError("provider unavailable")

    result = evaluate(
        [example((1, 2, 3, 4, 5, 6, 7, 0, 8))],
        APIErrorAgent(),
        num_rollouts=2,
    )

    assert len(result.episodes) == 2
    assert result.summary()["api_error"] == 2
    assert result.summary()["api_error_rate"] == 1.0
    assert result.summary()["mean_reward"] == 0.0


def test_deepseek_adapter_starts_a_fresh_request_each_turn() -> None:
    class Completions:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            response = type("Response", (), {})()
            response.choices = [
                type(
                    "Choice",
                    (),
                    {"message": type("Message", (), {"content": '{"tile": 8}'})()},
                )()
            ]
            return response

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client)
    agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))
    agent.next_action((1, 2, 3, 4, 5, 6, 0, 7, 8))
    assert len(completions.calls) == 2
    assert len(completions.calls[0]["messages"]) == 2
    assert completions.calls[0]["messages"][1] != completions.calls[1]["messages"][1]
    assert all(len(call["messages"]) == 2 for call in completions.calls)


def test_local_jsonl_eval_subset_is_supported(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "board": [1, 2, 3, 4, 5, 6, 7, 0, 8],
                "action_interface": "tile_id_v1",
                "optimal_actions": [8],
                "optimal_length": 1,
                "bucket": "easy",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tasks = load_examples(dataset=str(path))
    assert len(tasks) == 1
    assert tasks[0].board == (1, 2, 3, 4, 5, 6, 7, 0, 8)
    assert tasks[0].metadata["bucket"] == "easy"


def test_legacy_directional_dataset_rows_are_rejected(tmp_path) -> None:
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        json.dumps(
            {
                "board": [1, 2, 3, 4, 5, 6, 7, 0, 8],
                "optimal_moves": ["left"],
                "optimal_length": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="legacy optimal_moves is unsupported"):
        load_examples(dataset=str(path))


def test_local_parquet_eval_subset_is_supported(tmp_path) -> None:
    path = tmp_path / "eval.parquet"
    Dataset.from_list(
        [
            {
                "board": [1, 2, 3, 4, 5, 6, 7, 0, 8],
                "action_interface": "tile_id_v1",
                "optimal_actions": [8],
                "optimal_length": 1,
                "bucket": "easy",
            }
        ]
    ).to_parquet(str(path))

    tasks = load_examples(dataset=str(path))

    assert len(tasks) == 1
    assert tasks[0].board == (1, 2, 3, 4, 5, 6, 7, 0, 8)
    assert tasks[0].metadata["bucket"] == "easy"


def test_dotenv_key_is_supported(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("DEEPSEEK_API_KEY='from-dotenv'\n", encoding="utf-8")
    assert get_api_key(dotenv_path=dotenv) == "from-dotenv"


def test_environment_key_takes_precedence_over_dotenv(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("DEEPSEEK_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-environment")
    assert get_api_key(dotenv_path=dotenv) == "from-environment"


def test_solved_reward_is_bounded_and_efficiency_sensitive() -> None:
    assert solved_reward(4, 4) == 1.0
    assert solved_reward(4, 8) == 0.9
    assert solved_reward(4, 100) == 0.808


def test_exact_distance_is_goal_oriented() -> None:
    one_move = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    hardest = (8, 6, 7, 2, 5, 4, 3, 0, 1)
    assert exact_distance(GOAL) == 0
    assert exact_distance(one_move) == 1
    assert exact_distance(hardest) == 31


def test_distance_progress_rewards_progress_and_penalizes_backtracking() -> None:
    one_move = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    farther = (1, 2, 3, 4, 5, 6, 0, 7, 8)
    closer = distance_progress_reward(one_move, GOAL)
    farther_reward = distance_progress_reward(one_move, farther)
    assert closer == pytest.approx(0.25 / 31)
    assert farther_reward == pytest.approx(-0.25 / 31)
    assert closer + farther_reward == pytest.approx(0.0)


def test_solved_episode_includes_terminal_and_progress_rewards() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, SequenceAgent(['{"tile": 8}']))
    assert result.outcome == "solved"
    assert result.reward == pytest.approx(1.0 + 0.25 / 31)
    assert result.steps[0].terminal_reward == pytest.approx(1.0)
    assert result.steps[0].progress_reward == pytest.approx(0.25 / 31)
    assert result.steps[0].legal_tiles == (5, 7, 8)
    assert result.steps[0].tile == 8
    assert result.final_board == (1, 2, 3, 4, 5, 6, 7, 8, 0)


def test_serialized_steps_contain_no_directional_move_field() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, SequenceAgent(['{"tile": 8}']))
    assert "move" not in result.steps[0].to_dict()


def test_illegal_move_ends_immediately_with_negative_reward() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, SequenceAgent(['{"tile": 1}']))
    assert result.outcome == "illegal"
    assert result.reward == -1.0
    assert result.steps[0].progress_reward == 0.0
    assert result.steps[0].terminal_reward == -1.0
    assert result.steps[0].legal_tiles == (5, 7, 8)
    assert result.steps[0].tile == 1
    assert len(result.steps) == 1


def test_malformed_response_ends_immediately_with_negative_reward() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, SequenceAgent(["I choose left"]))
    assert result.outcome == "malformed"
    assert result.reward == -1.0
    assert result.steps[0].progress_reward == 0.0
    assert result.moves_taken == 0


def test_truncated_response_is_reported_separately() -> None:
    class TruncatedAgent:
        last_response_metadata = {"truncated": True}

        def next_action(self, board: tuple[int, ...]) -> str:
            return ""

    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, TruncatedAgent())
    assert result.outcome == "truncated"
    assert result.steps[0].status == "truncated"
    assert result.reward == -1.0
    assert result.steps[0].progress_reward == 0.0
    assert evaluate([task], TruncatedAgent()).summary()["truncated"] == 1


def test_multiple_rollouts_report_rollout_metrics_and_pass_at_k() -> None:
    class FirstFailsThenSolves:
        def __init__(self) -> None:
            self.calls = 0

        def next_action(self, board: tuple[int, ...]) -> str:
            self.calls += 1
            return "bad" if self.calls == 1 else '{"tile": 8}'

    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate([task], FirstFailsThenSolves(), num_rollouts=3)
    summary = result.summary()
    assert len(result.episodes) == 3
    assert summary["num_examples"] == 1
    assert summary["num_rollouts"] == 3
    assert summary["num_episodes"] == 3
    assert summary["solved"] == 2
    assert summary["malformed"] == 1
    assert summary["pass@k"] == 1.0
    assert [episode.rollout_id for episode in result.episodes] == [0, 1, 2]


def test_parallel_rollouts_preserve_order_and_isolate_agents() -> None:
    import threading

    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    lock = threading.Lock()
    created = 0

    class WorkerAgent:
        def __init__(self, worker_id: int) -> None:
            self.worker_id = worker_id
            self.last_response_metadata = {"worker_id": worker_id}

        def next_action(self, board: tuple[int, ...]) -> str:
            return '{"tile": 8}'

    def factory() -> WorkerAgent:
        nonlocal created
        with lock:
            worker_id = created
            created += 1
        return WorkerAgent(worker_id)

    result = evaluate(
        [task, task], num_rollouts=2, parallelism=2, agent_factory=factory
    )
    assert len(result.episodes) == 4
    assert [
        (episode.example.example_id, episode.rollout_id) for episode in result.episodes
    ] == [("test", 0), ("test", 1), ("test", 0), ("test", 1)]
    assert {
        episode.steps[0].response_metadata["worker_id"] for episode in result.episodes
    } == {0, 1, 2, 3}


def test_parallelism_requires_agent_factory() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    with pytest.raises(ValueError, match="agent_factory is required"):
        evaluate([task], SequenceAgent(['{"tile": 8}']), parallelism=2)


def test_num_rollouts_must_be_positive() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    with pytest.raises(ValueError, match="num_rollouts must be positive"):
        evaluate([task], SequenceAgent(['{"tile": 8}']), num_rollouts=0)


def test_valid_unsolved_trajectory_at_limit_gets_progress_and_timeout_penalty() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, SequenceAgent(['{"tile": 7}']), max_turns=1)
    assert result.outcome == "timeout"
    assert result.reward == pytest.approx(-0.25 - 0.25 / 31)
    assert result.steps[0].progress_reward == pytest.approx(-0.25 / 31)
    assert result.steps[0].terminal_reward == pytest.approx(-0.25)
    assert result.moves_taken == 1


def test_turn_limit_cannot_exceed_45() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    with pytest.raises(ValueError, match="between 1 and 45"):
        evaluate_episode(task, SequenceAgent([]), max_turns=46)


def test_harness_runs_ten_huggingface_dataset_examples(monkeypatch) -> None:
    from evaluation import dataset as dataset_module

    rows = [
        {
            "board": [1, 2, 3, 4, 5, 6, 7, 0, 8],
            "action_interface": "tile_id_v1",
            "optimal_actions": [8],
            "optimal_length": 1,
        }
    ] * 10
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: rows)
    tasks = load_examples(limit=10)

    # Use a fresh oracle that derives the next action from the current board by BFS;
    # this makes the smoke test independent of any model or credentials.
    from puzzle3.solver import solve

    class BfsOracle:
        def next_action(self, board: tuple[int, ...]) -> str:
            return json.dumps({"tile": solve(board)[0]})

    result = evaluate(tasks, BfsOracle())
    assert len(result.episodes) == 10
    assert result.summary()["solved"] == 10
    assert result.summary()["mean_reward"] == pytest.approx(1.0 + 0.25 / 31)
