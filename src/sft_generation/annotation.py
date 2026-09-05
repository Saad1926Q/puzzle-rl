"""Generate and validate rationales for verified trajectories."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from puzzle3.render import render
from sft_generation.client import OpenRouterTextClient, TextCompletion
from sft_generation.constants import ANNOTATION_PROMPT
from sft_generation.rollout import validate_trajectory


@dataclass(frozen=True)
class AnnotationConfig:
    """Settings shared by all annotation requests."""

    model: str
    base_url: str
    max_tokens: int = 512
    thinking: bool = True
    reasoning_effort: str | None = None
    temperature: float = 0.3
    top_p: float = 1.0
    upstream_providers: tuple[str, ...] = ()
    allow_fallbacks: bool = False
    require_parameters: bool = True
    data_collection: str = "deny"
    provider_retries: int = 2
    retry_delay: float = 1.0


def annotation_messages(
    trajectory: dict[str, Any], step_index: int, *, context_turns: int = 4
) -> list[dict[str, str]]:
    """Build one annotation request from a verified move."""

    validate_trajectory(trajectory)
    steps = trajectory["steps"]
    step = steps[step_index]
    start = max(0, step_index - context_turns)
    context = []
    for prior in steps[start:step_index]:
        context.append(
            f"Turn {prior['turn']}: {render(tuple(prior['board']))} -> "
            f"tile {prior['tile']} -> {render(tuple(prior['next_board']))}"
        )
    context_text = "\n".join(context) if context else "(no previous moves)"
    user = "\n".join(
        (
            "Goal board: 1 2 3 / 4 5 6 / 7 8 0",
            "Verified solving context:",
            context_text,
            f"Current board:\n{render(tuple(step['board']))}",
            f"Verified selected tile: {step['tile']}",
            f"Resulting board:\n{render(tuple(step['next_board']))}",
            "Explain this verified move.",
        )
    )
    return [
        {"role": "system", "content": ANNOTATION_PROMPT},
        {"role": "user", "content": user},
    ]


def clean_rationale(content: str) -> str:
    """Remove transport markup while preserving the generated explanation."""

    value = content.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
        if value.startswith("text"):
            value = value[4:].strip()
    if value.startswith("<think>") and value.endswith("</think>"):
        value = value[7:-8].strip()
    return " ".join(value.split())


def validate_annotation(
    annotation: dict[str, Any], trajectory: dict[str, Any], step_index: int
) -> None:
    """Validate annotation identity, mechanics, and concise wording."""

    validate_trajectory(trajectory)
    steps = trajectory["steps"]
    step = steps[step_index]
    if annotation.get("source_id") != trajectory["source_id"]:
        raise ValueError("annotation source_id does not match trajectory")
    if annotation.get("turn") != step["turn"]:
        raise ValueError("annotation turn does not match trajectory")
    if annotation.get("board") != step["board"]:
        raise ValueError("annotation board does not match trajectory")
    if annotation.get("tile") != step["tile"]:
        raise ValueError("annotation tile does not match trajectory")
    if annotation.get("next_board") != step["next_board"]:
        raise ValueError("annotation next_board does not match trajectory")
    rationale = annotation.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("annotation rationale is empty")
    words = rationale.split()
    if not 5 <= len(words) <= 60:
        raise ValueError("annotation rationale must contain 5 to 60 words")
    forbidden = ("optimal", "solver", "reward", "exact distance")
    lowered = rationale.lower()
    if any(term in lowered for term in forbidden):
        raise ValueError("annotation rationale contains forbidden information")


def annotate_step(
    trajectory: dict[str, Any],
    step_index: int,
    *,
    client: OpenRouterTextClient,
) -> dict[str, Any]:
    """Generate one validated rationale record."""

    step = trajectory["steps"][step_index]
    completion: TextCompletion = client.complete(annotation_messages(trajectory, step_index))
    annotation = {
        "source_id": trajectory["source_id"],
        "turn": step["turn"],
        "board": step["board"],
        "tile": step["tile"],
        "next_board": step["next_board"],
        "rationale": clean_rationale(completion.content),
        "valid": True,
        "response_metadata": completion.metadata,
    }
    validate_annotation(annotation, trajectory, step_index)
    return annotation


def annotation_futures(
    trajectories: Iterable[dict[str, Any]],
    *,
    api_key: str,
    config: AnnotationConfig,
    parallelism: int,
    skip_keys: set[tuple[str, int]] | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Annotate every missing move and retain invalid results for inspection."""

    skip_keys = skip_keys or set()
    results: dict[tuple[str, int], dict[str, Any]] = {}
    tasks: dict[Future[dict[str, Any]], tuple[dict[str, Any], int]] = {}
    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        for trajectory in trajectories:
            validate_trajectory(trajectory)
            for step_index, step in enumerate(trajectory["steps"]):
                key = (trajectory["source_id"], step["turn"])
                if key in skip_keys:
                    continue
                client = OpenRouterTextClient(
                    api_key=api_key,
                    model=config.model,
                    base_url=config.base_url,
                    thinking=config.thinking,
                    reasoning_effort=config.reasoning_effort,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    upstream_providers=config.upstream_providers,
                    allow_fallbacks=config.allow_fallbacks,
                    require_parameters=config.require_parameters,
                    data_collection=config.data_collection,
                    provider_retries=config.provider_retries,
                    retry_delay=config.retry_delay,
                )
                future = pool.submit(annotate_step, trajectory, step_index, client=client)
                tasks[future] = (trajectory, step_index)
        for future in as_completed(tasks):
            trajectory, step_index = tasks[future]
            step = trajectory["steps"][step_index]
            key = (trajectory["source_id"], step["turn"])
            try:
                annotation = future.result()
            except Exception as exc:
                annotation = {
                    "source_id": trajectory["source_id"],
                    "turn": step["turn"],
                    "board": step["board"],
                    "tile": step["tile"],
                    "next_board": step["next_board"],
                    "rationale": "",
                    "valid": False,
                    "error": str(exc),
                }
            results[key] = annotation
            if on_result is not None:
                on_result(annotation)
    return results
