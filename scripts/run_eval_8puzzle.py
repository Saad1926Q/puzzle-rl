#!/usr/bin/env python3
"""Evaluate a DeepSeek-compatible model on the 8-puzzle eval set.

Example:
    uv run python scripts/run_eval_8puzzle.py \
      --model deepseek-v4-flash \
      --output eval/results_8puzzle_deepseek_v4_flash.json

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.constants import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_THINKING,
    MAX_TURNS,
)
from evaluation.dataset import load_examples
from evaluation.evaluator import EvaluationResult, evaluate
from evaluation.model import DeepSeekAgent, get_api_key


def bounded_max_turns(value: str) -> int:
    turns = int(value)
    if not 1 <= turns <= MAX_TURNS:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_TURNS}")
    return turns


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="saad1926q/8-puzzle")
    parser.add_argument("--config", default="eval")
    parser.add_argument("--split", default="eval")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Evaluate only this many rows; default is the entire eval split",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--num-rollouts",
        type=positive_int,
        default=1,
        help="Independent rollouts per puzzle (default: 1)",
    )
    parser.add_argument("--max-turns", type=bounded_max_turns, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        help="Enable DeepSeek reasoning mode (default)",
    )
    thinking_group.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Disable reasoning and force the strict named tool call",
    )
    parser.set_defaults(thinking=DEFAULT_THINKING)
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results_8puzzle_deepseek_v4_flash.json"),
        help="JSON file for summary metrics",
    )
    parser.add_argument(
        "--save-trajectories",
        "--save_trajectories",
        dest="save_trajectories",
        action="store_true",
        help="Save complete episode/step traces to a separate JSON file",
    )
    return parser


def metadata(args: argparse.Namespace, actual_num_examples: int) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "num_examples": actual_num_examples,
        "num_rollouts": args.num_rollouts,
        "offset": args.offset,
        "max_turns": args.max_turns,
        "model": args.model,
        "base_url": args.base_url,
        "thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "save_trajectories": args.save_trajectories,
    }


def main() -> None:
    args = build_parser().parse_args()
    examples = load_examples(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        limit=args.num_examples,
        offset=args.offset,
    )

    api_key = get_api_key(args.api_key_env, args.dotenv)
    agent = DeepSeekAgent(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
    )

    result: EvaluationResult = evaluate(
        examples,
        agent,
        max_turns=args.max_turns,
        num_rollouts=args.num_rollouts,
    )
    run_metadata = metadata(args, len(examples))
    summary_output = {"metadata": run_metadata, "summary": result.summary()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path: Path | None = None
    if args.save_trajectories:
        trajectory_path = args.output.with_name(
            f"{args.output.stem}.trajectories{args.output.suffix or '.json'}"
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

    args.output.write_text(
        json.dumps(summary_output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result.summary(), indent=2))
    print(f"Wrote summary to {args.output}")
    if trajectory_path:
        print(f"Wrote trajectories to {trajectory_path}")


if __name__ == "__main__":
    main()
