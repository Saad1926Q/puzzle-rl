"""OpenRouter text generation for SFT annotations."""

from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Any, Sequence

from evaluation.clients.common import api_error_metadata, validate_reasoning_effort
from evaluation.constants import DEFAULT_OPENROUTER_BASE_URL
from evaluation.protocol import json_safe


@dataclass(frozen=True)
class TextCompletion:
    """One visible text completion and provider metadata."""

    content: str
    metadata: dict[str, Any]


class OpenRouterTextClient:
    """Request text completions without puzzle tools."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        thinking: bool = True,
        reasoning_effort: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 1.0,
        upstream_providers: Sequence[str] = (),
        allow_fallbacks: bool = False,
        require_parameters: bool = True,
        data_collection: str = "deny",
        provider_retries: int = 2,
        retry_delay: float = 1.0,
        client: Any | None = None,
    ) -> None:
        if reasoning_effort is not None:
            validate_reasoning_effort(reasoning_effort)
        if data_collection not in {"allow", "deny"}:
            raise ValueError("data_collection must be allow or deny")
        if provider_retries < 0 or retry_delay < 0:
            raise ValueError("retry settings must be non-negative")
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
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
        self.provider_retries = provider_retries
        self.retry_delay = retry_delay
        self.last_response_metadata: dict[str, Any] = {}

    def complete(self, messages: list[dict[str, Any]]) -> TextCompletion:
        """Return one visible completion for the supplied messages."""

        provider: dict[str, Any] = {
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
            "data_collection": self.data_collection,
        }
        if self.upstream_providers:
            provider["only"] = list(self.upstream_providers)
        request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_body": {
                "reasoning": (
                    {"enabled": True, "exclude": False}
                    if self.thinking and self.reasoning_effort is None
                    else {
                        "effort": self.reasoning_effort if self.thinking else "none",
                        "exclude": False,
                    }
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
                self.last_response_metadata = api_error_metadata(exc)
                raise
        try:
            choice = response.choices[0]
            message = choice.message
        except (AttributeError, IndexError, TypeError) as exc:
            self.last_response_metadata = {"status": "api_error", "error": str(exc)}
            raise RuntimeError("response did not contain a completion choice") from exc
        content = _text(getattr(message, "content", None)).strip()
        finish_reason = getattr(choice, "finish_reason", None)
        metadata = {
            "status": "truncated" if finish_reason == "length" else "complete",
            "finish_reason": finish_reason,
            "content_length": len(content),
            "reasoning_content_length": len(
                _text(getattr(message, "reasoning_content", None))
                or _text(getattr(message, "reasoning", None))
            ),
            "usage": json_safe(getattr(response, "usage", None)),
            "response_id": getattr(response, "id", None),
            "resolved_model": getattr(response, "model", None),
        }
        self.last_response_metadata = metadata
        if not content or metadata["status"] == "truncated":
            raise ValueError("annotation response was empty or truncated")
        return TextCompletion(content=content, metadata=metadata)


def _text(value: Any) -> str:
    """Extract visible text from an SDK content value."""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(block.get("text", "") for block in value if isinstance(block, dict))
    return ""


def _retryable(exc: Exception) -> bool:
    """Return whether an upstream request is safe to retry."""

    status = getattr(exc, "status_code", None)
    return status in {408, 409, 429} or isinstance(status, int) and status >= 500
