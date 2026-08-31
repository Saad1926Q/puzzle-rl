#!/usr/bin/env python3
"""Evaluate a model on the authoritative tile-action 8-puzzle set.

Example:
    uv run python scripts/run_eval_8puzzle.py \
      --provider qwen \
      --model Qwen/Qwen3.5-0.8B \
      --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.constants import (
    ACTION_INTERFACE,
    DEFAULT_API_KEY_ENV,
    DEFAULT_CROF_API_KEY_ENV,
    DEFAULT_CROF_BASE_URL,
    DEFAULT_CROF_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_GLM_API_KEY_ENV,
    DEFAULT_GLM_BASE_URL,
    DEFAULT_GLM_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_API_KEY_ENV,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENROUTER_API_KEY_ENV,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_QWEN_PRESENCE_PENALTY,
    DEFAULT_QWEN_REPETITION_PENALTY,
    DEFAULT_QWEN_TEMPERATURE,
    DEFAULT_QWEN_TOP_K,
    DEFAULT_QWEN_TOP_P,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_THINKING,
    DISTANCE_PROGRESS_WEIGHT,
    MAX_PUZZLE_DISTANCE,
    MAX_TURNS,
    REWARD_SCHEME,
)
from evaluation.dataset import load_examples
from evaluation.evaluator import EvaluationResult, evaluate
from evaluation.clients.crof import CrofAgent
from evaluation.clients.deepseek import DeepSeekAgent
from evaluation.clients.glm import GLMAgent
from evaluation.clients.openai_client import OpenAIAgent
from evaluation.clients.openrouter import OpenRouterAgent
from evaluation.clients.qwen import QwenAgent
from evaluation.protocol import get_api_key


type ProviderAgent = CrofAgent | DeepSeekAgent | GLMAgent | OpenAIAgent | OpenRouterAgent | QwenAgent


@dataclass(frozen=True)
class ProviderConfig:
    agent_class: type[ProviderAgent]
    default_model: str | None
    default_base_url: str
    default_api_key_env: str | None
    default_output: str
    default_thinking: bool


PROVIDERS = {
    "crof": ProviderConfig(
        CrofAgent,
        DEFAULT_CROF_MODEL,
        DEFAULT_CROF_BASE_URL,
        DEFAULT_CROF_API_KEY_ENV,
        "results_8puzzle_crof_glm_5_3_flash.json",
        DEFAULT_THINKING,
    ),
    "deepseek": ProviderConfig(
        DeepSeekAgent,
        DEFAULT_MODEL,
        DEFAULT_BASE_URL,
        DEFAULT_API_KEY_ENV,
        "results_8puzzle_deepseek_v4_flash.json",
        DEFAULT_THINKING,
    ),
    "glm": ProviderConfig(
        GLMAgent,
        DEFAULT_GLM_MODEL,
        DEFAULT_GLM_BASE_URL,
        DEFAULT_GLM_API_KEY_ENV,
        "results_8puzzle_glm_4_5_air.json",
        DEFAULT_THINKING,
    ),
    "openai": ProviderConfig(
        OpenAIAgent,
        DEFAULT_OPENAI_MODEL,
        DEFAULT_OPENAI_BASE_URL,
        DEFAULT_OPENAI_API_KEY_ENV,
        "results_8puzzle_openai.json",
        DEFAULT_THINKING,
    ),
    "openrouter": ProviderConfig(
        OpenRouterAgent,
        None,
        DEFAULT_OPENROUTER_BASE_URL,
        DEFAULT_OPENROUTER_API_KEY_ENV,
        "results_8puzzle_openrouter.json",
        DEFAULT_THINKING,
    ),
    "qwen": ProviderConfig(
        QwenAgent,
        DEFAULT_QWEN_MODEL,
        DEFAULT_QWEN_BASE_URL,
        None,
        "results_8puzzle_qwen3_5_0_8b.json",
        False,
    ),
}


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
        "--keep-history",
        action="store_true",
        help="Include the previous four completed board/action turns in each request",
    )
    parser.add_argument(
        "--keep-reasoning",
        action="store_true",
        help="Include saved reasoning in history; requires --keep-history",
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
def resolve_provider_args(args: argparse.Namespace) -> None:
    provider = PROVIDERS[args.provider]
    args.model = args.model or provider.default_model
    args.base_url = args.base_url or provider.default_base_url
    args.api_key_env = args.api_key_env or provider.default_api_key_env
    if args.provider == "openrouter" and args.model is None:
        raise ValueError("--model is required for --provider openrouter")
    if args.thinking is None:
        args.thinking = provider.default_thinking
    if args.reasoning_effort is None and args.provider != "openrouter":
        args.reasoning_effort = DEFAULT_REASONING_EFFORT


def metadata(args: argparse.Namespace, actual_num_examples: int) -> dict[str, Any]:
    result = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "num_examples": actual_num_examples,
        "num_rollouts": args.num_rollouts,
        "parallelism": args.parallelism,
        "offset": args.offset,
        "max_turns": args.max_turns,
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "keep_history": args.keep_history,
        "keep_reasoning": args.keep_reasoning,
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
                "openrouter_require_parameters": True,
                "openrouter_quantizations": args.openrouter_quantization,
                "openrouter_data_collection": args.openrouter_data_collection,
                "openrouter_distillable_only": args.openrouter_distillable_only,
            }
        )
    return result


def main() -> None:
    args = build_parser().parse_args()
    resolve_provider_args(args)
    if args.keep_reasoning and not args.keep_history:
        raise ValueError("--keep-reasoning requires --keep-history")
    provider = PROVIDERS[args.provider]
    if args.output is None:
        args.output = Path("eval") / provider.default_output
    examples = load_examples(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        limit=args.num_examples,
        offset=args.offset,
    )

    api_key = (
        get_api_key(args.api_key_env, args.dotenv)
        if args.api_key_env is not None
        else "not-required"
    )
    worker_local = threading.local()

    def agent_factory() -> ProviderAgent:
        agent = getattr(worker_local, "agent", None)
        if agent is None:
            agent_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "model": args.model,
                "base_url": args.base_url,
                "thinking": args.thinking,
                "reasoning_effort": args.reasoning_effort,
                "max_tokens": args.max_tokens,
            }
            if args.provider == "qwen":
                agent_kwargs.update(
                    {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "presence_penalty": args.presence_penalty,
                        "repetition_penalty": args.repetition_penalty,
                    }
                )
            elif args.provider == "openrouter":
                agent_kwargs.update(
                    {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "upstream_providers": args.openrouter_upstream,
                        "allow_fallbacks": args.openrouter_allow_fallbacks,
                        "require_parameters": True,
                        "data_collection": args.openrouter_data_collection,
                        "distillable_only": args.openrouter_distillable_only,
                        "quantizations": args.openrouter_quantization,
                    }
                )
            agent = provider.agent_class(**agent_kwargs)
            worker_local.agent = agent
        return agent

    result: EvaluationResult = evaluate(
        examples,
        max_turns=args.max_turns,
        num_rollouts=args.num_rollouts,
        parallelism=args.parallelism,
        keep_history=args.keep_history,
        keep_reasoning=args.keep_reasoning,
        agent_factory=agent_factory,
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
