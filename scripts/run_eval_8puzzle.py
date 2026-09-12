#!/usr/bin/env python3
"""Evaluate a model on the authoritative tile-action 8-puzzle set.

Example:
    uv run python scripts/run_eval_8puzzle.py \
      --provider qwen \
      --model Qwen/Qwen3.5-4B \
      --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from evaluation.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TURNS,
    DEFAULT_QWEN_PRESENCE_PENALTY,
    DEFAULT_QWEN_REPETITION_PENALTY,
    DEFAULT_QWEN_TEMPERATURE,
    DEFAULT_QWEN_TOP_K,
    DEFAULT_QWEN_TOP_P,
    MAX_TURNS,
)
from evaluation.dataset import load_examples
from evaluation.evaluator import evaluate
from evaluation.providers import PROVIDERS, ProviderSettings, create_agent_factory
from evaluation.reporting import metadata, write_evaluation_artifacts
from evaluation.results import EvaluationResult





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
    parser.add_argument(
        "--parallelism",
        type=positive_int,
        default=1,
        help="Concurrent rollout workers (default: 1)",
    )
    parser.add_argument(
        "--max-turns",
        type=bounded_max_turns,
        default=DEFAULT_MAX_TURNS,
        help="Maximum moves per episode (default: 45)",
    )
    parser.add_argument("--provider", choices=tuple(PROVIDERS), default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        help="Enable provider-supported reasoning mode",
    )
    thinking_group.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Disable provider-supported reasoning mode",
    )
    parser.set_defaults(thinking=None)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "max", "xhigh"),
        default=None,
        help="Reasoning effort override; omit to use the provider default",
    )
    parser.add_argument("--max-tokens", type=positive_int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--openrouter-upstream",
        action="append",
        default=[],
        metavar="PROVIDER",
        help="Restrict OpenRouter requests to this upstream provider; repeatable",
    )
    parser.add_argument(
        "--openrouter-allow-fallbacks",
        action="store_true",
        help="Allow OpenRouter to retry a different provider after a failed request",
    )
    parser.add_argument(
        "--openrouter-relax-parameters",
        action="store_true",
        help=(
            "Permit an upstream that does not support every requested parameter; "
            "OpenRouter may ignore unsupported parameters"
        ),
    )
    parser.add_argument(
        "--openrouter-quantization",
        action="append",
        default=[],
        metavar="QUANTIZATION",
        help="Restrict OpenRouter endpoints to this quantization; repeatable",
    )
    parser.add_argument(
        "--openrouter-data-collection",
        choices=("allow", "deny"),
        default="deny",
        help="OpenRouter upstream data-collection policy (default: deny)",
    )
    parser.add_argument(
        "--openrouter-distillable-only",
        action="store_true",
        help="Require OpenRouter endpoints that permit text distillation",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_QWEN_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_QWEN_TOP_P)
    parser.add_argument("--top-k", type=positive_int, default=DEFAULT_QWEN_TOP_K)
    parser.add_argument(
        "--presence-penalty", type=float, default=DEFAULT_QWEN_PRESENCE_PENALTY
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_QWEN_REPETITION_PENALTY,
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Include the previous four turns, including their reasoning",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
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







def main() -> None:
    args = build_parser().parse_args()
    settings = ProviderSettings.from_args(args)
    provider = PROVIDERS[settings.provider]
    keep_history = args.history
    if args.output is None:
        args.output = Path("eval") / provider.default_output
    examples = load_examples(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        limit=args.num_examples,
        offset=args.offset,
    )
    agent_factory = create_agent_factory(settings, args.dotenv)

    result: EvaluationResult = evaluate(
        examples,
        max_turns=args.max_turns,
        num_rollouts=args.num_rollouts,
        parallelism=args.parallelism,
        keep_history=keep_history,
    )
    run_metadata = metadata(args, len(examples), settings)
    trajectory_path = write_evaluation_artifacts(
        args.output,
        run_metadata=run_metadata,
        result=result,
        save_trajectories=args.save_trajectories,
    )
    print(json.dumps(result.summary(), indent=2))
    print(f"Wrote summary to {args.output}")
    if trajectory_path:
        print(f"Wrote trajectories to {trajectory_path}")


if __name__ == "__main__":
    main()
