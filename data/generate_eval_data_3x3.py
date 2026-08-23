import argparse
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from datasets import Dataset

from puzzle3.board import GOAL, _get_opposite, apply_move, legal_moves

BUCKETS = (
    ("easy", 1, 5),
    ("medium", 6, 15),
    ("hard", 16, 31),
)


def _compute_bucket_counts(num_tasks: int) -> list[tuple[str, int, int, int]]:
    """Allocate an equal number of examples to each difficulty bucket."""

    base_count, remainder = divmod(num_tasks, len(BUCKETS))
    return [
        (label, min_depth, max_depth, base_count + (index < remainder))
        for index, (label, min_depth, max_depth) in enumerate(BUCKETS)
    ]


def _enumerate_from_goal() -> dict[tuple[int, ...], list[str]]:
    """Return every reachable 8-puzzle state with an optimal path from GOAL to it."""

    queue = deque([GOAL])
    paths = {GOAL: []}

    while queue:
        board = queue.popleft()
        for move in legal_moves(board):
            nxt = apply_move(board, move)
            if nxt in paths:
                continue
            paths[nxt] = [*paths[board], move]
            queue.append(nxt)

    return paths


def _solution_from_goal_path(path_from_goal: list[str]) -> list[str]:
    """Convert GOAL->board path to board->GOAL solution moves."""

    return [_get_opposite(move) for move in reversed(path_from_goal)]


def _make_eval_record(
    board: tuple[int, ...], bucket: str, path_from_goal: list[str]
) -> dict[str, Any]:
    optimal_moves = _solution_from_goal_path(path_from_goal)
    return {
        "board": list(board),
        "scramble_depth": len(optimal_moves),
        "bucket": bucket,
        "split": "eval_3x3",
        "optimal_moves": optimal_moves,
        "optimal_length": len(optimal_moves),
    }


def _select_state_quantiles(
    choices: list[tuple[tuple[int, ...], list[str]]],
    count: int,
    rng: random.Random,
) -> list[tuple[tuple[int, ...], list[str]]]:
    """Select quantiles from states ordered by optimal distance."""

    if count == 0:
        return []
    if count > len(choices):
        raise ValueError(
            f"requested {count} puzzles but only {len(choices)} are available"
        )

    by_depth: dict[int, list[tuple[tuple[int, ...], list[str]]]] = defaultdict(list)
    for choice in choices:
        by_depth[len(choice[1])].append(choice)
    ordered: list[tuple[tuple[int, ...], list[str]]] = []
    for depth in sorted(by_depth):
        depth_choices = by_depth[depth]
        rng.shuffle(depth_choices)
        ordered.extend(depth_choices)

    if count == 1:
        indices = [len(ordered) // 2]
    else:
        indices = [
            round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)
        ]
    return [ordered[index] for index in indices]


def generate_eval_candidates(
    num_tasks: int, rng: random.Random
) -> list[dict[str, Any]]:
    paths = _enumerate_from_goal()
    by_bucket: dict[str, list[tuple[tuple[int, ...], list[str]]]] = defaultdict(list)

    for board, path in paths.items():
        depth = len(path)
        for label, min_depth, max_depth in BUCKETS:
            if min_depth <= depth <= max_depth:
                by_bucket[label].append((board, path))
                break

    records = []
    for label, _min_depth, _max_depth, count in _compute_bucket_counts(num_tasks):
        choices = _select_state_quantiles(by_bucket[label], count, rng)
        records.extend(_make_eval_record(board, label, path) for board, path in choices)

    return records


def write_eval_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def write_eval_parquet(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a fixed mixed-difficulty 3x3 / 8-puzzle eval set."
    )
    parser.add_argument("--num-tasks", type=int, default=45)
    parser.add_argument(
        "--output", type=Path, default=Path("data/eval_puzzles_3x3_45.jsonl")
    )
    parser.add_argument(
        "--parquet-output",
        type=Path,
        default=Path("data/eval_puzzles_3x3_45.parquet"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_tasks <= 0:
        raise ValueError("num_tasks must be positive")

    records = generate_eval_candidates(args.num_tasks, random.Random(args.seed))
    write_eval_jsonl(records, args.output)
    write_eval_parquet(records, args.parquet_output)
    print(f"Generated {len(records)} 3x3 eval puzzles at {args.output}")
    print(f"Wrote Parquet eval data at {args.parquet_output}")


if __name__ == "__main__":
    main()
