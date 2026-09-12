"""CLI run metadata and JSON artifact reporting for puzzle evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.constants import (
    ACTION_INTERFACE,
    DISTANCE_PROGRESS_WEIGHT,
    MAX_PUZZLE_DISTANCE,
    REWARD_SCHEME,
)
from evaluation.providers import ProviderSettings
from evaluation.results import EvaluationResult


def metadata(
    args: argparse.Namespace,
    actual_num_examples: int,
    settings: ProviderSettings,
) -> dict[str, Any]:
    """Build the run metadata embedded in evaluation JSON artifacts."""

    result = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "num_examples": actual_num_examples,
        "num_rollouts": args.num_rollouts,
        "parallelism": args.parallelism,
        "offset": args.offset,
        "max_turns": args.max_turns,
        "provider": settings.provider,
        "model": settings.model,
        "base_url": settings.base_url,
        "thinking": settings.thinking,
        "reasoning_effort": settings.reasoning_effort,
        "max_tokens": settings.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "history": args.history,
        "save_trajectories": args.save_trajectories,
        "action_interface": ACTION_INTERFACE,
        "reward_scheme": REWARD_SCHEME,
        "distance_progress_weight": DISTANCE_PROGRESS_WEIGHT,
        "max_puzzle_distance": MAX_PUZZLE_DISTANCE,
    }
    if args.provider == "openrouter":
        result.update(
            {
                "openrouter_upstreams": args.openrouter_upstream,
                "openrouter_allow_fallbacks": args.openrouter_allow_fallbacks,
                "openrouter_require_parameters": not args.openrouter_relax_parameters,
                "openrouter_quantizations": args.openrouter_quantization,
                "openrouter_data_collection": args.openrouter_data_collection,
                "openrouter_distillable_only": args.openrouter_distillable_only,
            }
        )
    return result


def write_evaluation_artifacts(
    output: Path,
    *,
    run_metadata: dict[str, Any],
    result: EvaluationResult,
    save_trajectories: bool,
) -> Path | None:
    """Write the summary and optional trajectory artifacts, returning its path."""

    summary_output = {"metadata": run_metadata, "summary": result.summary()}
    output.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path: Path | None = None
    if save_trajectories:
        trajectory_path = output.with_name(
            f"{output.stem}.trajectories{output.suffix or '.json'}"
        )
        trajectory_output = {
            "metadata": run_metadata,
            "summary": result.summary(),
            "episodes": [episode.to_dict() for episode in result.episodes],
        }
        trajectory_path.write_text(
            json.dumps(trajectory_output, indent=2) + "\n", encoding="utf-8"
        )
        summary_output["trajectory_file"] = str(trajectory_path)

    output.write_text(json.dumps(summary_output, indent=2) + "\n", encoding="utf-8")
    return trajectory_path
