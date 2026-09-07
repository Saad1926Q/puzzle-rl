from __future__ import annotations

import pytest

from evaluation.clients.crof import CrofAgent
from evaluation.clients.deepseek import DeepSeekAgent
from evaluation.clients.glm import GLMAgent
from evaluation.clients.openai_client import OpenAIAgent
from evaluation.clients.openrouter import OpenRouterAgent
from evaluation.clients.qwen import QwenAgent
from evaluation.dataset import PuzzleExample
from evaluation.evaluator import evaluate
from evaluation.protocol import HistoryTurn, parse_tile
from puzzle3.board import GOAL
from tests.evaluation._fakes import (
    chat_client,
    chat_response,
    responses_client,
    responses_response,
    tool_call,
)


def example(board: tuple[int, ...], optimal_length: int = 1) -> PuzzleExample:
    return PuzzleExample("test", board, tuple(), optimal_length, {})


def test_crof_uses_standard_chat_parameters_and_reasoning() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return chat_response(
                tool_calls=tool_call('{"tile": 8}'),
                reasoning_content="Move tile 8 into the blank.",
            )

    completions = Completions()
    client = chat_client(completions)
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
            return chat_response(tool_calls=tool_call())

    completions = Completions()
    client = chat_client(completions)
    CrofAgent(api_key="not-used", client=client, thinking=False).next_action(
        (1, 2, 3, 4, 5, 6, 7, 0, 8)
    )

    assert "reasoning_effort" not in completions.kwargs

def test_openrouter_sends_reproducible_routing_and_records_metadata() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            response = chat_response(
                tool_calls=tool_call('{"tile": 8}'),
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
    client = chat_client(completions)
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
            return chat_response(tool_calls=tool_call())

    completions = Completions()
    client = chat_client(completions)
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
            return chat_response(tool_calls=tool_call())

    completions = Completions()
    client = chat_client(completions)
    OpenRouterAgent(
        api_key="not-used",
        model="test/model",
        client=client,
    ).next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))

    assert completions.kwargs["extra_body"]["reasoning"] == {
        "enabled": True,
        "exclude": False,
    }

def test_openrouter_empty_choices_becomes_api_error_episode() -> None:
    class Completions:
        def create(self, **kwargs):
            response = chat_response(tool_calls=tool_call())
            response.choices = None
            return response

    completions = Completions()
    client = chat_client(completions)
    agent = OpenRouterAgent(api_key="not-used", model="test/model", client=client)

    result = evaluate([example((1, 2, 3, 4, 5, 6, 7, 0, 8))], agent)

    assert result.summary()["api_error"] == 1
    assert agent.last_response_metadata["status"] == "api_error"
    assert agent.last_response_metadata["error_type"] == "RuntimeError"

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
            return chat_response(tool_calls=tool_call())

    completions = Completions()
    client = chat_client(completions)
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
            return chat_response(tool_calls=tool_call())

    completions = Completions()
    client = chat_client(completions)
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert parse_tile(agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))) == 8
    assert completions.kwargs["tool_choice"]["function"]["name"] == "slide_tile"
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert agent.last_response_metadata["status"] == "tool_call"

def test_glm_tool_call_uses_zai_parameters() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return chat_response(tool_calls=tool_call('{"tile":3}'))

    completions = Completions()
    client = chat_client(completions)
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
            return chat_response(tool_calls=tool_call())

    completions = Completions()
    client = chat_client(completions)
    agent = GLMAgent(api_key="not-used", client=client, thinking=False)
    agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}

def test_qwen_uses_native_tool_call_and_sampling_parameters() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return chat_response(
                tool_calls=tool_call('{"tile":8}'),
                reasoning="reason about the board",
            )

    completions = Completions()
    client = chat_client(completions)
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
            return chat_response(tool_calls=tool_call())

    completions = Completions()
    client = chat_client(completions)

    QwenAgent(client=client).next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))

    assert completions.kwargs["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }

def test_openai_responses_tool_call_uses_responses_parameters() -> None:
    class Responses:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return responses_response(arguments='{"tile":6}')

    responses = Responses()
    client = responses_client(responses)
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
            return responses_response(status="incomplete")

    client = responses_client(Responses())
    agent = OpenAIAgent(api_key="not-used", client=client)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["truncated"] is True
    assert agent.last_response_metadata["status"] == "truncated"
    assert agent.last_response_metadata["incomplete_reason"] == "max_output_tokens"

def test_deepseek_thinking_mode_reads_tool_call_and_reasoning_metadata() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return chat_response(
                tool_calls=tool_call('{"tile":3}'),
                reasoning_content="reason about the board",
            )

    completions = Completions()
    client = chat_client(completions)
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=True)
    assert parse_tile(agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))) == 3
    assert completions.kwargs["tool_choice"] == "auto"
    assert agent.last_response_metadata["reasoning_content_length"] > 0

def test_deepseek_multiple_tool_calls_are_malformed() -> None:
    class Completions:
        def create(self, **kwargs):
            return chat_response(
                tool_calls=tool_call('{"tile":8}') + tool_call('{"tile":3}')
            )

    completions = Completions()
    client = chat_client(completions)
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["tool_call_count"] == 2
    assert agent.last_response_metadata["status"] == "malformed"

def test_deepseek_truncated_tool_call_is_not_accepted() -> None:
    class Completions:
        def create(self, **kwargs):
            return chat_response(
                tool_calls=tool_call('{"tile":8}'), finish_reason="length"
            )

    completions = Completions()
    client = chat_client(completions)
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["truncated"] is True
    assert agent.last_response_metadata["status"] == "truncated"

def test_deepseek_empty_or_truncated_response_is_diagnosed() -> None:
    class Completions:
        def create(self, **kwargs):
            return chat_response(
                content="",
                tool_calls=None,
                finish_reason="length",
                reasoning_content="partial",
            )

    completions = Completions()
    client = chat_client(completions)
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=True)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["status"] == "truncated"
    assert agent.last_response_metadata["truncated"] is True
    assert agent.last_response_metadata["reasoning_content_length"] == 7

def test_deepseek_malformed_tool_arguments_are_diagnosed() -> None:
    class Completions:
        def create(self, **kwargs):
            return chat_response(tool_calls=tool_call('{"tile":0}'))

    completions = Completions()
    client = chat_client(completions)
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["tool_name"] == "slide_tile"
    assert agent.last_response_metadata["status"] == "malformed"

def test_deepseek_provider_error_is_not_relabelled_as_malformed() -> None:
    class Completions:
        def create(self, **kwargs):
            raise RuntimeError("provider unavailable")

    completions = Completions()
    client = chat_client(completions)
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))
    assert agent.last_response_metadata["status"] == "api_error"

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
    client = chat_client(completions)
    agent = DeepSeekAgent(api_key="not-used", client=client)
    agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))
    agent.next_action((1, 2, 3, 4, 5, 6, 0, 7, 8))
    assert len(completions.calls) == 2
    assert len(completions.calls[0]["messages"]) == 2
    assert completions.calls[0]["messages"][1] != completions.calls[1]["messages"][1]
    assert all(len(call["messages"]) == 2 for call in completions.calls)

