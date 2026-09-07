from __future__ import annotations

import pytest

from sft_generation.training import (
    is_decision_record,
    render_training_record,
    validate_decision_record,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        del kwargs
        return [1, 2] if add_generation_prompt else [1, 2, 3, 4]


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
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }


def test_training_helpers_reject_puzzle_validation_rows() -> None:
    record = decision_record()
    record["metadata"] = {"record_type": "puzzle"}
    assert not is_decision_record(record)
    with pytest.raises(ValueError, match="record_type"):
        validate_decision_record(record)
