"""Training-time validation and rendering helpers for SFT records."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import json
from typing import Any
DECISION_RECORD_TYPE = "decision"


def is_decision_record(record: dict[str, Any]) -> bool:
    """Return whether a dataset row is a supervised decision example."""
    metadata = record.get("metadata")
    return isinstance(metadata, dict) and metadata.get("record_type") == DECISION_RECORD_TYPE


def validate_decision_record(record: dict[str, Any]) -> None:
    """Validate the observable SFT contract before tokenization."""
    if not is_decision_record(record):
        raise ValueError("SFT training rows must have metadata.record_type='decision'")
    prompt = record.get("prompt")
    completion = record.get("completion")
    tools = record.get("tools")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("SFT decision prompt must be a non-empty message list")
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValueError("SFT decision completion must contain one message")
    target = completion[0]
    if target.get("role") != "assistant":
        raise ValueError("SFT decision completion must be an assistant message")
    tool_calls = target.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise ValueError("SFT decision completion must contain one tool call")
    if not isinstance(tools, list) or len(tools) != 1:
        raise ValueError("SFT decision must contain one tool schema")


def normalize_tool_arguments(record: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI wire-format argument strings to template mappings."""
    normalized = copy.deepcopy(record)
    for messages_key in ("prompt", "completion"):
        for message in normalized.get(messages_key, []):
            for tool_call in message.get("tool_calls", []):
                function = tool_call.get("function", tool_call)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        parsed = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise ValueError("tool-call arguments are not valid JSON") from exc
                    if not isinstance(parsed, dict):
                        raise ValueError("tool-call arguments must decode to an object")
                    function["arguments"] = parsed
    return normalized


def _token_ids(value: Any) -> list[int]:
    """Normalize tokenizer output from list or tensor-like values."""
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def render_training_record(tokenizer: Any, record: dict[str, Any]) -> dict[str, int]:
    """Render one record and return prompt/completion token lengths."""
    validate_decision_record(record)
    normalized = normalize_tool_arguments(record)
    prompt = normalized["prompt"]
    completion = normalized["completion"]
    tools = normalized["tools"]
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            prompt,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    full_ids = _token_ids(
        tokenizer.apply_chat_template(
            prompt + completion,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("tokenized prompt is not a prefix of the full SFT sequence")
    completion_length = len(full_ids) - len(prompt_ids)
    if completion_length <= 0:
        raise ValueError("SFT completion produced no trainable tokens")
    return {
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": completion_length,
        "total_tokens": len(full_ids),
    }
