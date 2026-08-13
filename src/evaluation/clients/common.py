"""Shared client helpers."""

from __future__ import annotations

from typing import Any

from evaluation.constants import DEFAULT_REASONING_EFFORT


def validate_reasoning_effort(reasoning_effort: str) -> None:
    if reasoning_effort not in {"low", "medium", "high", "max"}:
        raise ValueError("reasoning_effort must be low, medium, high, or max")


def api_error_metadata(exc: Exception) -> dict[str, Any]:
    return {
        "status": "api_error",
        "error_type": type(exc).__name__,
        "error": str(exc)[:1000],
    }


def empty_reasoning_metadata(reasoning: str = "") -> dict[str, Any]:
    return {
        "reasoning_content_length": len(reasoning),
        "reasoning_content": reasoning,
        "reasoning_preview": reasoning[:1000],
    }
