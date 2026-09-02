"""OpenRouter rollout client owned by the GEPA experiment."""

from __future__ import annotations

import json
from time import sleep
from typing import Any, Sequence

from prompt_optimization.eval.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OPENROUTER_BASE_URL,
    SLIDE_TILE_TOOL,
)
from prompt_optimization.eval.protocol import HistoryTurn, build_messages, json_safe
from puzzle3.board import Board


class OpenRouterAgent:
    """Candidate-prompt OpenRouter client for isolated GEPA rollouts."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        system_prompt: str,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        thinking: bool = True,
        reasoning_effort: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 1.0,
        top_p: float = 1.0,
        upstream_providers: Sequence[str] = (),
        allow_fallbacks: bool = False,
        data_collection: str = "deny",
        distillable_only: bool = False,
        quantizations: Sequence[str] = (),
        provider_retries: int = 2,
        retry_delay: float = 1.0,
        request_timeout: float = 120.0,
        client: Any | None = None,
    ) -> None:
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=request_timeout,
                default_headers={
                    "X-OpenRouter-Metadata": "enabled",
                    "X-OpenRouter-Title": "puzzle-rl-gepa",
                },
            )
        if reasoning_effort not in {None, "minimal", "low", "medium", "high", "max", "xhigh"}:
            raise ValueError("invalid reasoning_effort")
        if data_collection not in {"allow", "deny"}:
            raise ValueError("data_collection must be allow or deny")
        if provider_retries < 0 or retry_delay < 0:
            raise ValueError("retry settings must be non-negative")
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.upstream_providers = tuple(upstream_providers)
        self.allow_fallbacks = allow_fallbacks
        self.data_collection = data_collection
        self.distillable_only = distillable_only
        self.quantizations = tuple(quantizations)
        self.provider_retries = provider_retries
        self.retry_delay = retry_delay
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
            "require_parameters": True,
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
            "messages": build_messages(
                board, self.system_prompt, history, include_reasoning=include_reasoning
            ),
            "tools": [SLIDE_TILE_TOOL],
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_body": {
                "reasoning": (
                    {"enabled": True, "exclude": False}
                    if self.thinking and self.reasoning_effort is None
                    else {"effort": self.reasoning_effort if self.thinking else "none", "exclude": False}
                ),
                "provider": provider,
            },
        }
        for attempt in range(self.provider_retries + 1):
            try:
                response = self.client.chat.completions.create(**request)
                break
            except Exception as exc:
                if attempt < self.provider_retries and _retryable(exc):
                    sleep(self.retry_delay * (attempt + 1))
                    continue
                self.last_response_metadata = _api_error_metadata(exc)
                raise
        result, metadata = _parse_response(response)
        metadata.update({
            "response_id": _field(response, "id"),
            "resolved_model": _field(response, "model"),
            "openrouter_metadata": json_safe(_field(response, "openrouter_metadata")),
        })
        self.last_response_metadata = metadata
        return result


def _parse_response(response: Any) -> tuple[str, dict[str, Any]]:
    try:
        choice = response.choices[0]
        message = choice.message
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenRouter response did not contain a completion choice") from exc
    calls = getattr(message, "tool_calls", None) or []
    tile = None
    if len(calls) == 1:
        function = getattr(calls[0], "function", None)
        if getattr(function, "name", None) == "slide_tile":
            try:
                payload = json.loads(getattr(function, "arguments", ""))
                candidate = payload.get("tile") if isinstance(payload, dict) else None
                tile = candidate if type(candidate) is int and 1 <= candidate <= 8 else None
            except json.JSONDecodeError:
                pass
    content = getattr(message, "content", None)
    text = content if isinstance(content, str) else ""
    reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
    reasoning = reasoning if isinstance(reasoning, str) else ""
    finish_reason = getattr(choice, "finish_reason", None)
    truncated = finish_reason == "length"
    metadata = {
        "status": "truncated" if truncated else ("tool_call" if tile is not None else "malformed"),
        "finish_reason": finish_reason,
        "content": text,
        "content_length": len(text),
        "reasoning_content": reasoning,
        "reasoning_content_length": len(reasoning),
        "reasoning_preview": reasoning[:1000],
        "reasoning_details": json_safe(getattr(message, "reasoning_details", None)),
        "truncated": truncated,
        "tool_call_count": len(calls),
        "usage": json_safe(getattr(response, "usage", None)),
    }
    return ("" if truncated else json.dumps({"tile": tile}) if tile is not None else text), metadata


def _field(response: Any, name: str) -> Any:
    value = getattr(response, name, None)
    if value is not None:
        return value
    extra = getattr(response, "model_extra", None)
    return extra.get(name) if isinstance(extra, dict) else None


def _api_error_metadata(exc: Exception) -> dict[str, Any]:
    return {"status": "api_error", "error_type": type(exc).__name__, "error": str(exc)[:1000]}


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status in {408, 409, 429} or isinstance(status, int) and status >= 500
