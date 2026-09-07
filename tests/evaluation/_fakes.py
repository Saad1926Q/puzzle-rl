from __future__ import annotations

from types import SimpleNamespace


def chat_client(completions: object) -> SimpleNamespace:
    """Build the minimal OpenAI-compatible chat client used by provider tests."""
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def responses_client(responses: object) -> SimpleNamespace:
    """Build the minimal OpenAI Responses client used by provider tests."""
    return SimpleNamespace(responses=responses)


def chat_response(
    *,
    content: str = "",
    tool_calls: object | None = None,
    finish_reason: str = "tool_calls",
    reasoning: str | None = None,
    reasoning_content: str | None = None,
    reasoning_details: object | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
        reasoning_details=reasoning_details,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=None
    )


def tool_call(arguments: str = '{"tile":8}', name: str = "slide_tile") -> list[SimpleNamespace]:
    return [SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))]


def responses_response(
    *, arguments: str = '{"tile":8}', status: str = "completed"
) -> SimpleNamespace:
    incomplete = SimpleNamespace(reason="max_output_tokens")
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call", name="slide_tile", arguments=arguments
            )
        ],
        output_text="",
        status=status,
        incomplete_details=incomplete if status == "incomplete" else None,
        usage=None,
    )
