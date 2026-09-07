from __future__ import annotations

import pytest

from sft_generation.training import (
    is_decision_record,
    normalize_tool_arguments,
    render_training_record,
    validate_decision_record,
)

class FakeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        del kwargs, add_generation_prompt
        return list(range(1, len(messages) + 3))


def decision_record() -> dict:
    return {
        "prompt": [{"role": "user", "content": "board"}],
        "completion": [
            {
                "role": "assistant",
                "content": "<think>move</think>",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "slide_tile",
                            "arguments": '{"tile": 7}',
                        },
                    }
                ],
            }
        ],
        "metadata": {"record_type": "decision"},
        "tools": [{"type": "function", "function": {"name": "slide_tile"}}],
    }


def test_training_helpers_accept_decision_rows_only() -> None:
    record = decision_record()
    assert is_decision_record(record)
    validate_decision_record(record)
    assert render_training_record(FakeTokenizer(), record) == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


def test_tool_arguments_are_normalized_for_chat_templates() -> None:
    record = decision_record()
    normalized = normalize_tool_arguments(record)
    assert normalized["completion"][0]["tool_calls"][0]["function"]["arguments"] == {"tile": 7}
    assert record["completion"][0]["tool_calls"][0]["function"]["arguments"] == '{"tile": 7}'


def test_training_helpers_reject_puzzle_validation_rows() -> None:
    record = decision_record()
    record["metadata"] = {"record_type": "puzzle"}
    assert not is_decision_record(record)
    with pytest.raises(ValueError, match="record_type"):
        validate_decision_record(record)
