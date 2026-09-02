"""Compact, deterministic trajectory feedback for GEPA reflection."""

from __future__ import annotations

from collections import Counter
from typing import Any

from prompt_optimization.eval.evaluator import EpisodeResult
from puzzle3.solver import exact_distance

_MAX_REASONING_CHARS = 1_200


def _completion_tokens(metadata: dict[str, Any] | None) -> int:
    usage = (metadata or {}).get("usage")
    if not isinstance(usage, dict):
        return 0
    value = usage.get("completion_tokens", 0)
    return value if type(value) is int and value >= 0 else 0


def _reasoning_excerpts(episode: EpisodeResult, distances: list[int]) -> list[dict[str, Any]]:
    """Select high-signal model reasoning rather than serializing whole episodes."""
    if not episode.steps:
        return []


    selected = {0, len(episode.steps) - 1}
    for index, step in enumerate(episode.steps):
        if index and step.tile == episode.steps[index - 1].tile:
            selected.add(index)
        if distances[index + 1] == min(distances):
            selected.add(index)

    excerpts = []
    for index in sorted(selected):
        if index < 0:
            continue
        step = episode.steps[index]
        reasoning = (step.response_metadata or {}).get("reasoning_content", "")
        if not isinstance(reasoning, str) or not reasoning.strip():
            continue
        excerpts.append(
            {
                "turn": step.turn,
                "distance_before": distances[index],
                "distance_after": distances[index + 1],
                "tile": step.tile,
                "completion_tokens": _completion_tokens(step.response_metadata),
                "text": reasoning[:_MAX_REASONING_CHARS],
            }
        )
    return excerpts


def episode_reflection_record(episode: EpisodeResult) -> dict[str, Any]:
    """Return reflection evidence without disclosing any optimal action sequence."""

    boards = [episode.example.board]
    boards.extend(step.next_board or step.board for step in episode.steps)
    distances = [exact_distance(board) for board in boards]
    progress = [step.progress_reward for step in episode.steps]
    tiles = [step.tile for step in episode.steps]
    completion_tokens = [_completion_tokens(step.response_metadata) for step in episode.steps]
    repeated_boards = sum(count - 1 for count in Counter(boards).values() if count > 1)
    immediate_reversals = sum(
        current == previous
        for previous, current in zip(tiles, tiles[1:], strict=False)
        if current is not None and previous is not None
    )
    longest_nonprogress = 0
    current_nonprogress = 0
    for reward in progress:
        if reward <= 0:
            current_nonprogress += 1
            longest_nonprogress = max(longest_nonprogress, current_nonprogress)
        else:
            current_nonprogress = 0

    action_trace = [
        {
            "turn": step.turn,
            "tile": step.tile,
            "legal_tiles": list(step.legal_tiles),
            "distance_before": distances[index],
            "distance_after": distances[index + 1],
            "progress_reward": step.progress_reward,
            "status": step.status,
        }
        for index, step in enumerate(episode.steps)
    ]
    best_distance = min(distances)
    best_turn = distances.index(best_distance)
    diagnostic = (
        f"Outcome: {episode.outcome}. Reward: {episode.reward:.6f}. "
        f"Moves: {episode.moves_taken}; optimal depth: {episode.example.optimal_length}. "
        f"Best exact distance: {best_distance} at turn {best_turn}; "
        f"final exact distance: {distances[-1]}. "
        f"Distance-reducing actions: {sum(reward > 0 for reward in progress)}; "
        f"distance-increasing actions: {sum(reward < 0 for reward in progress)}; "
        f"immediate reversals: {immediate_reversals}; repeated boards: {repeated_boards}; "
        f"longest non-progress run: {longest_nonprogress}."
    )
    if episode.outcome == "api_error":
        diagnostic += " This is infrastructure failure, not strategy feedback."

    return {
        "Inputs": {
            "example_id": episode.example.example_id,
            "initial_board": list(episode.example.board),
            "optimal_depth": episode.example.optimal_length,
            "bucket": episode.example.metadata.get("bucket"),
        },
        "Generated Outputs": {
            "outcome": episode.outcome,
            "reward": episode.reward,
            "moves_taken": episode.moves_taken,
            "final_board": list(episode.final_board),
            "final_distance": distances[-1],
            "action_trace": action_trace,
            "completion_tokens": sum(completion_tokens),
            "reasoning_excerpts": _reasoning_excerpts(episode, distances),
        },
        "Feedback": diagnostic,
    }
