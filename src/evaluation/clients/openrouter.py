"""OpenRouter Chat Completions client for reproducible teacher evaluations."""

from __future__ import annotations

from typing import Any, Sequence

from evaluation.clients.common import (
    api_error_metadata,
    parse_chat_response,
    validate_reasoning_effort,
)
from evaluation.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_THINKING,
    SLIDE_TILE_TOOL,
)
from evaluation.protocol import HistoryTurn, build_chat_completion_messages, json_safe
from puzzle3.board import Board


class OpenRouterAgent:
    """One-request-per-state OpenRouter adapter using Chat Completions."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        thinking: bool = DEFAULT_THINKING,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 1.0,
        top_p: float = 1.0,
        upstream_providers: Sequence[str] = (),
        allow_fallbacks: bool = False,
        require_parameters: bool = True,
        data_collection: str = "deny",
        distillable_only: bool = False,
        quantizations: Sequence[str] = (),
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers={
                    "X-OpenRouter-Metadata": "enabled",
                    "X-OpenRouter-Title": "puzzle-rl",
                },
            )
        validate_reasoning_effort(reasoning_effort)
        if data_collection not in {"allow", "deny"}:
            raise ValueError("data_collection must be allow or deny")
        self.client = client
        self.model = model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.upstream_providers = tuple(upstream_providers)
        self.allow_fallbacks = allow_fallbacks
        self.require_parameters = require_parameters
        self.data_collection = data_collection
        self.distillable_only = distillable_only
        self.quantizations = tuple(quantizations)
        self.last_response_metadata: dict[str, Any] = {}

    def next_action(
        self,
        board: Board,
        history: Sequence[HistoryTurn] = (),
        *,
        include_reasoning: bool = False,
    ) -> str:
        provider: dict[str, Any] = {
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
            "data_collection": self.data_collection,
        }
        if self.upstream_providers:
            provider["only"] = list(self.upstream_providers)
        if self.quantizations:
            provider["quantizations"] = list(self.quantizations)
        if self.distillable_only:
            provider["enforce_distillable_text"] = True

        request = {
            "model": self.model,
            "messages": build_chat_completion_messages(
                board, history, include_reasoning=include_reasoning
            ),
            "tools": [SLIDE_TILE_TOOL],
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_body": {
                "reasoning": {
                    "effort": self.reasoning_effort if self.thinking else "none",
                    "exclude": False,
                },
                "provider": provider,
            },
        }
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            self.last_response_metadata = api_error_metadata(exc)
            raise
        try:
            result, response_metadata = parse_chat_response(
                response,
                provider="OpenRouter",
                truncated_reasons={"length"},
            )
        except RuntimeError:
            self.last_response_metadata = {
                "status": "invalid_response",
                "error": "response did not contain a completion choice",
            }
            raise
        response_metadata.update(
            {
                "response_id": _response_field(response, "id"),
                "resolved_model": _response_field(response, "model"),
                "openrouter_metadata": json_safe(
                    _response_field(response, "openrouter_metadata")
                ),
            }
        )
        self.last_response_metadata = response_metadata
        return result


def _response_field(response: Any, name: str) -> Any:
    """Read standard or OpenRouter extension fields from an SDK response."""
    value = getattr(response, name, None)
    if value is not None:
        return value
    model_extra = getattr(response, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(name)
    return None
