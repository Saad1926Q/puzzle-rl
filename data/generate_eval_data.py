import argparse
import json
import random
from typing import Any

from puzzle.scramble import scramble_board
from puzzle.solver import solve

EVAL_BUCKETS = (
    ("shallow", 4, 10, 0.2),
    ("medium", 15, 25, 0.4),
    ("deep", 60, 80, 0.4),
)


def _compute_bucket_counts(num_tasks: int) -> list[tuple[str, int, int, int]]:
    """
    Convert the fixed eval bucket ratios into exact counts for num_tasks.
    """

    shallow_count = num_tasks * 20 // 100
    medium_count = num_tasks * 40 // 100
    deep_count = num_tasks - shallow_count - medium_count

    return [
        ("shallow", 4, 10, shallow_count),
        ("medium", 15, 25, medium_count),
        ("deep", 60, 80, deep_count),
    ]


def _make_eval_record(
    board: tuple[int, ...], depth: int, bucket: str, optimal_moves: list[str]
) -> dict[str, Any]:
    """
    Build one eval record with depth bucket and optimal-solution metadata.
    """

    return {
        "board": board,
        "scramble_depth": depth,
        "bucket": bucket,
        "split": "eval",
        "optimal_moves": optimal_moves,
        "optimal_length": len(optimal_moves),
    }


def _generate_candidates_for_bucket(
    label: str,
    min_depth: int,
    max_depth: int,
    count: int,
    rng: random.Random,
    seen_boards: set,
) -> list[dict[str, Any]]:
    """
    Generate unique eval puzzles within one bucket range and annotate them with IDA* solutions.
    """

    generated_count = 0
    records = []

    while generated_count < count:
        depth = rng.randint(min_depth, max_depth)
        board = scramble_board(depth=depth, rng=rng)

        if board in seen_boards:
            continue

        seen_boards.add(board)
        optimal_moves = solve(board)
        records.append(_make_eval_record(board, depth, label, optimal_moves))
        generated_count += 1

    return records


def generate_eval_candidates(
    num_tasks: int, rng: random.Random
) -> list[dict[str, Any]]:
    """
    Generate a mixed-difficulty fixed eval set using the configured bucket ratios.
    """

    seen_boards = set()
    all_records = []

    for label, min_depth, max_depth, count in _compute_bucket_counts(num_tasks):
        generated_records = _generate_candidates_for_bucket(
            label=label,
            min_depth=min_depth,
            max_depth=max_depth,
            count=count,
            rng=rng,
            seen_boards=seen_boards,
        )
        all_records.extend(generated_records)

    return all_records


def write_eval_jsonl(records: list[dict[str, Any]], output_path: str) -> None:
    """
    Write the eval puzzle set to a JSONL file.
    """

    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def main() -> None:
    """
    Parse CLI arguments and generate a fixed mixed-difficulty eval set.
    """

    parser = argparse.ArgumentParser(
        description="Generate a fixed mixed-difficulty 15-puzzle eval set."
    )
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.num_tasks <= 0:
        raise ValueError("num_tasks must be positive")

    rng = random.Random(args.seed)
    records = generate_eval_candidates(args.num_tasks, rng)
    write_eval_jsonl(records, args.output)

    print(f"Generated {len(records)} eval puzzles at {args.output}")


if __name__ == "__main__":
    main()
