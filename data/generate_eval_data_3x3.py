import argparse
import json
import random
from collections import defaultdict, deque
from typing import Any

from puzzle3.board import GOAL, _get_opposite, apply_move, legal_moves

BUCKETS = (
    ("easy", 1, 5, 0.3),
    ("medium", 6, 15, 0.4),
    ("hard", 16, 31, 0.3),
)


def _compute_bucket_counts(num_tasks: int) -> list[tuple[str, int, int, int]]:
    easy_count = num_tasks * 30 // 100
    medium_count = num_tasks * 40 // 100
    hard_count = num_tasks - easy_count - medium_count
    return [
        ("easy", 1, 5, easy_count),
        ("medium", 6, 15, medium_count),
        ("hard", 16, 31, hard_count),
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


def _make_eval_record(board: tuple[int, ...], bucket: str, path_from_goal: list[str]) -> dict[str, Any]:
    optimal_moves = _solution_from_goal_path(path_from_goal)
    return {
        "board": list(board),
        "scramble_depth": len(optimal_moves),
        "bucket": bucket,
        "split": "eval_3x3",
        "optimal_moves": optimal_moves,
        "optimal_length": len(optimal_moves),
    }


def generate_eval_candidates(num_tasks: int, rng: random.Random) -> list[dict[str, Any]]:
    paths = _enumerate_from_goal()
    by_bucket: dict[str, list[tuple[tuple[int, ...], list[str]]]] = defaultdict(list)

    for board, path in paths.items():
        depth = len(path)
        for label, min_depth, max_depth, _ratio in BUCKETS:
            if min_depth <= depth <= max_depth:
                by_bucket[label].append((board, path))
                break

    records = []
    for label, _min_depth, _max_depth, count in _compute_bucket_counts(num_tasks):
        choices = list(by_bucket[label])
        if count > len(choices):
            raise ValueError(f"requested {count} {label} puzzles but only {len(choices)} are available")
        rng.shuffle(choices)
        records.extend(_make_eval_record(board, label, path) for board, path in choices[:count])

    return records


def write_eval_jsonl(records: list[dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fixed mixed-difficulty 3x3 / 8-puzzle eval set.")
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/eval_puzzles_3x3_100.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_tasks <= 0:
        raise ValueError("num_tasks must be positive")

    records = generate_eval_candidates(args.num_tasks, random.Random(args.seed))
    write_eval_jsonl(records, args.output)
    print(f"Generated {len(records)} 3x3 eval puzzles at {args.output}")


if __name__ == "__main__":
    main()
