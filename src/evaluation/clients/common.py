"""Shared client helpers."""

from __future__ import annotations

import json
from typing import Any

from evaluation.protocol import extract_chat_tool_tile, json_safe


def validate_reasoning_effort(reasoning_effort: str) -> None:
    if reasoning_effort not in {"minimal", "low", "medium", "high", "max", "xhigh"}:
        raise ValueError(
            "reasoning_effort must be minimal, low, medium, high, max, or xhigh"
        )


def api_error_metadata(exc: Exception) -> dict[str, Any]:
    return {
        "status": "api_error",
        "error_type": type(exc).__name__,
        "status_code": getattr(exc, "status_code", None),
        "error": str(exc)[:1000],
        "body": json_safe(getattr(exc, "body", None)),
    }


def empty_reasoning_metadata(reasoning: str = "") -> dict[str, Any]:
    return {
        "reasoning_content_length": len(reasoning),
        "reasoning_content": reasoning,
        "reasoning_preview": reasoning[:1000],
    }


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return ""


def parse_chat_response(
    response: Any,
    *,
    provider: str,
    truncated_reasons: set[str],
) -> tuple[str, dict[str, Any]]:
    """Normalize one OpenAI-compatible Chat Completions response."""

    try:
        choice = response.choices[0]
        message = choice.message
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"{provider} response did not contain a completion choice"
        ) from exc

    content = _text_content(getattr(message, "content", None))
    reasoning = _text_content(
        getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
    )
    tile, tool_metadata = extract_chat_tool_tile(message)
    finish_reason = getattr(choice, "finish_reason", None)
    truncated = finish_reason in truncated_reasons
    metadata = {
        "status": "truncated"
        if truncated
        else ("tool_call" if tile is not None else "malformed"),
        "finish_reason": finish_reason,
        "content": content,
        "content_length": len(content),
        **empty_reasoning_metadata(reasoning),
        "truncated": truncated,
        **tool_metadata,
        "usage": json_safe(getattr(response, "usage", None)),
    }
    canonical_response = json.dumps({"tile": tile}) if tile is not None else content
    return ("" if truncated else canonical_response), metadata
