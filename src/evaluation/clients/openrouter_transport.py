"""Shared OpenRouter transport policy for API clients."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import sleep
from typing import Any, TypeVar


Response = TypeVar("Response")


def create_openrouter_client(
    *,
    api_key: str,
    base_url: str,
    default_headers: Mapping[str, str] | None = None,
) -> Any:
    """Construct the OpenAI-compatible client used for OpenRouter requests."""
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    if default_headers is not None:
        kwargs["default_headers"] = dict(default_headers)
    return OpenAI(**kwargs)


def openrouter_extra_body(
    *,
    thinking: bool,
    reasoning_effort: str | None,
    allow_fallbacks: bool,
    require_parameters: bool,
    data_collection: str,
    upstream_providers: tuple[str, ...],
    quantizations: tuple[str, ...] = (),
    distillable_only: bool = False,
) -> dict[str, Any]:
    """Build OpenRouter's reasoning and reproducible provider-routing body."""
    provider: dict[str, Any] = {
        "allow_fallbacks": allow_fallbacks,
        "require_parameters": require_parameters,
        "data_collection": data_collection,
    }
    if upstream_providers:
        provider["only"] = list(upstream_providers)
    if quantizations:
        provider["quantizations"] = list(quantizations)
    if distillable_only:
        provider["enforce_distillable_text"] = True

    return {
        "reasoning": (
            {"enabled": True, "exclude": False}
            if thinking and reasoning_effort is None
            else {
                "effort": reasoning_effort if thinking else "none",
                "exclude": False,
            }
        ),
        "provider": provider,
    }


def execute_openrouter_request(
    request: Callable[[], Response],
    *,
    provider_retries: int,
    retry_delay: float,
    is_retryable: Callable[[Exception], bool],
) -> Response:
    """Run an OpenRouter request with the configured linear retry policy."""
    for attempt in range(provider_retries + 1):
        try:
            return request()
        except Exception as exc:
            if attempt < provider_retries and is_retryable(exc):
                sleep(retry_delay * (attempt + 1))
                continue
            raise
    raise AssertionError("unreachable")


def is_retryable_transport_error(exc: Exception) -> bool:
    """Identify transient OpenRouter transport failures."""
    status_code = getattr(exc, "status_code", None)
    return status_code in {408, 409, 429} or (
        isinstance(status_code, int) and status_code >= 500
    )
