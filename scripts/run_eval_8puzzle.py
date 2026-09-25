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
from evaluation.board_representations import BOARD_REPRESENTATIONS
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
from evaluation.results import EpisodeResult, EvaluationResult, episode_from_dict





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


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
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
    parser.add_argument("--top-k", type=nonnegative_int, default=DEFAULT_QWEN_TOP_K)
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
        "--board-representation",
        choices=BOARD_REPRESENTATIONS,
        default="grid",
        help="Board serialization used in model prompts (default: grid)",
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

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Append completed episodes to this JSONL file for resuming",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume unfinished episodes from --checkpoint-path",
    )
    return parser



CHECKPOINT_VERSION = 1
CHECKPOINT_SETTINGS = (
    "dataset",
    "config",
    "split",
    "num_examples",
    "offset",
    "num_rollouts",
    "max_turns",
    "provider",
    "model",
    "base_url",
    "thinking",
    "reasoning_effort",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "repetition_penalty",
    "history",
    "board_representation",
)


def checkpoint_settings(run_metadata: dict[str, object]) -> dict[str, object]:
    return {key: run_metadata[key] for key in CHECKPOINT_SETTINGS}


def load_checkpoint(
    path: Path,
    *,
    expected_settings: dict[str, object],
    examples: list[object],
) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], EpisodeResult]]]:

    examples_by_id = {example.example_id: example for example in examples}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"checkpoint is empty: {path}")
    header = json.loads(lines[0])
    if (
        header.get("type") != "metadata"
        or header.get("version") != CHECKPOINT_VERSION
        or header.get("settings") != expected_settings
    ):
        raise ValueError(f"checkpoint settings do not match this evaluation: {path}")

    completed_keys: set[tuple[int, int]] = set()
    episodes: list[tuple[tuple[int, int], EpisodeResult]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        record = json.loads(line)
        if record.get("type") != "episode":
            raise ValueError(f"invalid checkpoint record at {path}:{line_number}")
        key = (int(record["example_index"]), int(record["rollout_id"]))
        if key in completed_keys:
            raise ValueError(f"duplicate checkpoint episode at {path}:{line_number}")
        example_index, rollout_id = key
        if not 0 <= example_index < len(examples):
            raise ValueError(f"checkpoint example index is out of range: {key}")
        if not 0 <= rollout_id < int(expected_settings["num_rollouts"]):
            raise ValueError(f"checkpoint rollout ID is out of range: {key}")
        episode = episode_from_dict(record["episode"], examples_by_id)
        if episode.example.example_id != examples[example_index].example_id:
            raise ValueError(f"checkpoint example does not match its index: {key}")
        if episode.rollout_id != rollout_id:
            raise ValueError(f"checkpoint rollout does not match its key: {key}")
        completed_keys.add(key)
        episodes.append((key, episode))
    return completed_keys, episodes

def main() -> None:
    args = build_parser().parse_args()
    if args.resume and args.checkpoint_path is None:
        raise ValueError("--resume requires --checkpoint-path")

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
    run_metadata = metadata(args, len(examples), settings)

    completed_keys: set[tuple[int, int]] = set()
    checkpoint_episodes: list[tuple[tuple[int, int], EpisodeResult]] = []
    checkpoint_file = None
    if args.checkpoint_path is not None:
        settings_for_checkpoint = checkpoint_settings(run_metadata)
        if args.resume:
            completed_keys, checkpoint_episodes = load_checkpoint(
                args.checkpoint_path,
                expected_settings=settings_for_checkpoint,
                examples=examples,
            )
            checkpoint_file = args.checkpoint_path.open("a", encoding="utf-8")
        else:
            if args.checkpoint_path.exists():
                raise FileExistsError(
                    f"checkpoint already exists: {args.checkpoint_path}; "
                    "pass --resume to continue it"
                )
            args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_file = args.checkpoint_path.open("w", encoding="utf-8")
            checkpoint_file.write(
                json.dumps(
                    {
                        "type": "metadata",
                        "version": CHECKPOINT_VERSION,
                        "settings": settings_for_checkpoint,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            checkpoint_file.flush()

    def on_episode_complete(
        example_index: int, rollout_id: int, episode: EpisodeResult
    ) -> None:
        if checkpoint_file is None:
            return
        checkpoint_file.write(
            json.dumps(
                {
                    "type": "episode",
                    "example_index": example_index,
                    "rollout_id": rollout_id,
                    "episode": episode.to_dict(),
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        checkpoint_file.flush()

    try:
        new_result: EvaluationResult = evaluate(
            examples,
            max_turns=args.max_turns,
            num_rollouts=args.num_rollouts,
            parallelism=args.parallelism,
            keep_history=keep_history,
            agent_factory=agent_factory,
            completed_keys=completed_keys,
            on_episode_complete=(
                on_episode_complete if args.checkpoint_path is not None else None
            ),
        )
    finally:
        if checkpoint_file is not None:
            checkpoint_file.close()

    if args.checkpoint_path is None:
        result = new_result
    else:
        episodes_by_key = dict(checkpoint_episodes)
        example_indices = {
            example.example_id: index for index, example in enumerate(examples)
        }
        for episode in new_result.episodes:
            key = (example_indices[episode.example.example_id], episode.rollout_id)
            episodes_by_key[key] = episode
        episodes = [
            episodes_by_key[(example_index, rollout_id)]
            for example_index in range(len(examples))
            for rollout_id in range(args.num_rollouts)
            if (example_index, rollout_id) in episodes_by_key
        ]
        result = EvaluationResult(episodes, num_rollouts=args.num_rollouts)

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
