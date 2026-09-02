#!/usr/bin/env python3
"""Optimize an OpenRouter 8-puzzle strategy prompt with GEPA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.constants import DEFAULT_MAX_TOKENS, DEFAULT_MAX_TURNS
from prompt_optimization.dataset import DEFAULT_GEPA_CONFIG, load_gepa_splits
from prompt_optimization.runner import OpenRouterRolloutConfig, run_gepa_optimization


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _max_turns(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > DEFAULT_MAX_TURNS:
        raise argparse.ArgumentTypeError(f"must be at most {DEFAULT_MAX_TURNS}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="saad1926q/8-puzzle")
    parser.add_argument("--config", default=DEFAULT_GEPA_CONFIG)
    parser.add_argument("--task-model", required=True)
    parser.add_argument(
        "--reflection-model", default="openrouter/z-ai/glm-5.3-flash"
    )
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--openrouter-upstream", action="append", default=[])
    parser.add_argument("--openrouter-quantization", action="append", default=[])
    parser.add_argument("--openrouter-allow-fallbacks", action="store_true")
    parser.add_argument(
        "--openrouter-data-collection", choices=("allow", "deny"), default="deny"
    )
    parser.add_argument("--openrouter-distillable-only", action="store_true")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "max", "xhigh"),
        default=None,
    )
    parser.add_argument("--max-tokens", type=_positive_int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-turns", type=_max_turns, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--keep-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallelism", type=_positive_int, default=8)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--reflection-minibatch-size", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="GEPA run directory; reuse to resume"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    splits = load_gepa_splits(dataset=args.dataset, config=args.config)
    rollout = OpenRouterRolloutConfig(
        model=args.task_model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        upstream_providers=tuple(args.openrouter_upstream),
        allow_fallbacks=args.openrouter_allow_fallbacks,
        data_collection=args.openrouter_data_collection,
        distillable_only=args.openrouter_distillable_only,
        quantizations=tuple(args.openrouter_quantization),
        max_turns=args.max_turns,
        keep_history=args.keep_history,
        keep_reasoning=args.keep_reasoning,
        parallelism=args.parallelism,
        request_timeout=args.request_timeout,
    )
    result = run_gepa_optimization(
        splits=splits,
        rollout=rollout,
        reflection_model=args.reflection_model,
        output_dir=args.output_dir,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    print(f"Wrote GEPA artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
