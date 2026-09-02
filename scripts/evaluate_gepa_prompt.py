#!/usr/bin/env python3
"""Evaluate a frozen GEPA strategy with the isolated prompt-optimization stack."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from prompt_optimization.dataset import DEFAULT_GEPA_CONFIG, load_gepa_splits
from prompt_optimization.eval.client import OpenRouterAgent
from prompt_optimization.eval.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OPENROUTER_API_KEY_ENV,
    DEFAULT_OPENROUTER_BASE_URL,
    MAX_TURNS,
    build_candidate_system_prompt,
)
from prompt_optimization.eval.evaluator import EpisodeResult, evaluate_episode
from prompt_optimization.eval.protocol import get_api_key
from prompt_optimization.runner import OpenRouterRolloutConfig


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not parsed > 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--strategy-prompt-file", type=Path, required=True)
    result.add_argument("--task-model", required=True)
    result.add_argument("--dataset", default="saad1926q/8-puzzle")
    result.add_argument("--config", default=DEFAULT_GEPA_CONFIG)
    result.add_argument("--split", choices=("train", "validation", "test"), default="test")
    result.add_argument("--api-key-env", default=DEFAULT_OPENROUTER_API_KEY_ENV)
    result.add_argument("--base-url", default=DEFAULT_OPENROUTER_BASE_URL)
    result.add_argument("--openrouter-upstream", action="append", default=[])
    result.add_argument("--openrouter-quantization", action="append", default=[])
    result.add_argument("--openrouter-allow-fallbacks", action="store_true")
    result.add_argument(
        "--openrouter-data-collection", choices=("allow", "deny"), default="deny"
    )
    result.add_argument("--openrouter-distillable-only", action="store_true")
    result.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--reasoning-effort", default=None)
    result.add_argument("--max-tokens", type=positive_int, default=DEFAULT_MAX_TOKENS)
    result.add_argument("--max-turns", type=positive_int, default=MAX_TURNS)
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--top-p", type=float, default=1.0)
    result.add_argument("--num-rollouts", type=positive_int, default=4)
    result.add_argument("--parallelism", type=positive_int, default=4)
    result.add_argument("--request-timeout", type=positive_float, default=120.0)
    result.add_argument("--keep-history", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--keep-reasoning", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def episode_dict(episode: EpisodeResult) -> dict[str, object]:
    return {
        "id": episode.example.example_id,
        "rollout_id": episode.rollout_id,
        "initial_board": list(episode.example.board),
        "optimal_length": episode.example.optimal_length,
        "outcome": episode.outcome,
        "reward": episode.reward,
        "moves_taken": episode.moves_taken,
        "final_board": list(episode.final_board),
        "steps": [
            {
                "turn": step.turn,
                "board": list(step.board),
                "legal_tiles": list(step.legal_tiles),
                "raw_response": step.raw_response,
                "tile": step.tile,
                "next_board": list(step.next_board) if step.next_board else None,
                "status": step.status,
                "response_metadata": step.response_metadata,
                "reward": step.reward,
                "progress_reward": step.progress_reward,
                "terminal_reward": step.terminal_reward,
            }
            for step in episode.steps
        ],
    }


def main() -> None:
    args = parser().parse_args()
    if args.max_turns > MAX_TURNS:
        raise ValueError(f"--max-turns must be at most {MAX_TURNS}")
    if args.keep_reasoning and not args.keep_history:
        raise ValueError("--keep-reasoning requires --keep-history")
    strategy = args.strategy_prompt_file.read_text(encoding="utf-8").strip()
    prompt = build_candidate_system_prompt(strategy)
    splits = load_gepa_splits(dataset=args.dataset, config=args.config)
    examples = getattr(splits, args.split)
    api_key = get_api_key(args.api_key_env)
    config = OpenRouterRolloutConfig(
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

    def rollout(index: int, example) -> EpisodeResult:
        agent = OpenRouterAgent(
            api_key=api_key,
            model=config.model,
            system_prompt=prompt,
            base_url=config.base_url,
            thinking=config.thinking,
            reasoning_effort=config.reasoning_effort,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            upstream_providers=config.upstream_providers,
            allow_fallbacks=config.allow_fallbacks,
            data_collection=config.data_collection,
            distillable_only=config.distillable_only,
            quantizations=config.quantizations,
            request_timeout=config.request_timeout,
        )
        result = evaluate_episode(
            example,
            agent,
            max_turns=config.max_turns,
            keep_history=config.keep_history,
            keep_reasoning=config.keep_reasoning,
        )
        result.rollout_id = index
        return result

    jobs = [(rollout_id, example) for example in examples for rollout_id in range(args.num_rollouts)]
    episodes: list[EpisodeResult | None] = [None] * len(jobs)
    with (
        ThreadPoolExecutor(max_workers=min(args.parallelism, len(jobs))) as executor,
        tqdm(total=len(jobs), desc="Puzzle rollouts", unit="episode") as progress,
    ):
        future_indices: dict[Future[EpisodeResult], int] = {
            executor.submit(rollout, *job): index for index, job in enumerate(jobs)
        }
        for future in as_completed(future_indices):
            episodes[future_indices[future]] = future.result()
            progress.update(1)
    completed = [episode for episode in episodes if episode is not None]
    if len(completed) != len(jobs):
        raise RuntimeError("not all puzzle rollouts produced a result")
    solved = [episode for episode in completed if episode.outcome == "solved"]
    summary = {
        "num_episodes": len(completed),
        "solved": len(solved),
        "solved_rate": len(solved) / len(completed),
        "mean_reward": sum(episode.reward for episode in completed) / len(completed),
    }
    output = {
        "metadata": {
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "strategy_prompt_file": str(args.strategy_prompt_file),
            "rollout": config.__dict__,
        },
        "summary": summary,
        "episodes": [episode_dict(episode) for episode in completed],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
