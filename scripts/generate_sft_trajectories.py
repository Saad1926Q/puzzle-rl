"""Generate replay-verified teacher trajectories for SFT."""

from __future__ import annotations

import argparse
from pathlib import Path


from evaluation.dataset import DEFAULT_DATASET, load_examples
from evaluation.protocol import DEFAULT_API_KEY_ENV
from sft_generation.constants import DEFAULT_MAX_TOKENS, DEFAULT_MAX_TURNS, DEFAULT_TEACHER_MODEL
from sft_generation.rollout import (
    RolloutConfig,
    RolloutFailure,
    rollout_futures,
    select_successful_rollout,
    trajectory_record,
)
from sft_generation.storage import append_jsonl, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse trajectory-generation options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default="sft-source")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument(
        "--retry-skipped",
        action="store_true",
        help="retry source boards previously recorded as skipped",
    )
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--num-rollouts", type=int, default=2)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--upstream-provider", action="append", default=[])
    parser.add_argument("--allow-fallbacks", action="store_true")
    parser.add_argument("--allow-parameters", action="store_true")
    parser.add_argument("--data-collection", choices=("allow", "deny"), default="deny")
    parser.add_argument("--provider-retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_generation"))
    return parser.parse_args()


def existing_ids(path: Path) -> set[str]:
    """Return source IDs already written to an artifact."""

    return {str(record["source_id"]) for record in read_jsonl(path) if "source_id" in record}


def main() -> None:
    """Generate one verified trajectory per solvable source board."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    solved_path = args.output_dir / "solved_trajectories.jsonl"
    skipped_path = args.output_dir / "skipped_puzzles.jsonl"
    solved = existing_ids(solved_path)
    skipped = existing_ids(skipped_path)
    examples = load_examples(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        limit=args.num_examples,
        offset=args.offset,
        require_optimal_actions=False,
    )
    pending = [
        example
        for example in examples
        if example.example_id not in solved
        and (args.retry_skipped or example.example_id not in skipped)
    ]
    from evaluation.protocol import get_api_key

    config = RolloutConfig(
        model=args.model,
        base_url=args.base_url,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        thinking=not args.no_thinking,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        top_p=args.top_p,
        upstream_providers=tuple(args.upstream_provider),
        allow_fallbacks=args.allow_fallbacks,
        require_parameters=not args.allow_parameters,
        data_collection=args.data_collection,
        provider_retries=args.provider_retries,
        retry_delay=args.retry_delay,
    )
    solved_count = 0
    skipped_count = 0
    grouped = rollout_futures(
        pending,
        num_rollouts=args.num_rollouts,
        parallelism=args.parallelism,
        api_key=get_api_key(args.api_key_env),
        config=config,
    )
    for example in pending:
        outcomes = grouped[example.example_id]
        winner = select_successful_rollout(
            episode for episode in outcomes if not isinstance(episode, RolloutFailure)
        )
        if winner is not None:
            record = trajectory_record(winner)
            record["teacher_model"] = args.model
            append_jsonl(solved_path, record)
            solved_count += 1
            continue
        append_jsonl(
            skipped_path,
            {
                "source_id": example.example_id,
                "initial_board": list(example.board),
                "rollout_outcomes": [
                    {
                        "rollout_id": item.rollout_id,
                        "outcome": "exception" if isinstance(item, RolloutFailure) else item.outcome,
                        **({"error": item.error} if isinstance(item, RolloutFailure) else {}),
                    }
                    for item in sorted(outcomes, key=lambda value: value.rollout_id)
                ],
            },
        )
        skipped_count += 1
    if args.retry_skipped:
        current_solved = existing_ids(solved_path)
        remaining_skipped: dict[str, dict] = {}
        for record in read_jsonl(skipped_path):
            source_id = str(record.get("source_id", ""))
            if source_id and source_id not in current_solved:
                remaining_skipped[source_id] = record
        write_jsonl(skipped_path, remaining_skipped.values())
    write_json(
        args.output_dir / "generation_metadata.json",
        {
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "offset": args.offset,
            "num_examples": args.num_examples,
            "model": args.model,
            "num_rollouts": args.num_rollouts,
            "parallelism": args.parallelism,
            "retry_skipped": args.retry_skipped,
            "max_tokens": args.max_tokens,
            "thinking": not args.no_thinking,
            "reasoning_effort": args.reasoning_effort,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    )
    print(f"processed={len(pending)} solved={solved_count} skipped={skipped_count}")
    print(f"solved trajectories saved at {solved_path}")
    print(f"skipped puzzles saved at {skipped_path}")
    print(f"generation metadata saved at {args.output_dir / 'generation_metadata.json'}")


if __name__ == "__main__":
    main()
