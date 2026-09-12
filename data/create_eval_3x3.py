"""Generate the fixed 62-puzzle 8-puzzle evaluation dataset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.generation import (
    enumerate_from_goal,
    make_eval_record,
    write_eval_jsonl,
    write_eval_parquet,
)
from puzzle3.board import Board, GOAL, adjacent_tiles, slide_tile

DEPTHS = tuple(range(1, 32))
PUZZLES_PER_DEPTH = 2
EXPECTED_REACHABLE_STATES = 181_440
EXPECTED_MAX_DEPTH = 31
EXPECTED_RECORDS = len(DEPTHS) * PUZZLES_PER_DEPTH

DEFAULT_EXCLUSIONS = (
    Path("data/sft_validation_puzzles_3x3_depths_6_10_10.jsonl"),
    Path("data/sft_source_3x3_depths_10_19_200.jsonl"),
    Path("data/sft_source_3x3_depths_12_16_200.jsonl"),
    Path("data/sft_source_3x3_depths_12_16_1000.jsonl"),
    Path("data/sft_source_3x3_depths_12_16_1500.jsonl"),
)


def _bucket_for_depth(depth: int) -> str:
    if depth <= 10:
        return "easy"
    if depth <= 20:
        return "medium"
    return "hard"


def load_excluded_boards(paths: list[Path]) -> set[Board]:
    """Load boards reserved by other local datasets."""

    excluded: set[Board] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"exclusion dataset does not exist: {path}")
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                board = tuple(record.get("board", ()))
                if len(board) != len(GOAL) or set(board) != set(GOAL):
                    raise ValueError(f"invalid board at {path}:{line_number}")
                excluded.add(board)
    return excluded


def generate_eval_candidates(
    rng: random.Random,
    excluded_boards: set[Board] | None = None,
) -> list[dict[str, Any]]:
    """Sample two unseen boards at every exact distance from the goal."""

    excluded_boards = excluded_boards or set()
    paths = enumerate_from_goal()
    if len(paths) != EXPECTED_REACHABLE_STATES:
        raise RuntimeError(
            f"expected {EXPECTED_REACHABLE_STATES} reachable states, got {len(paths)}"
        )
    if max(map(len, paths.values())) != EXPECTED_MAX_DEPTH:
        raise RuntimeError(f"expected maximum depth {EXPECTED_MAX_DEPTH}")

    boards_by_depth: dict[int, list[Board]] = defaultdict(list)
    for board, path in paths.items():
        depth = len(path)
        if depth in DEPTHS and board not in excluded_boards:
            boards_by_depth[depth].append(board)

    records: list[dict[str, Any]] = []
    for depth in DEPTHS:
        candidates = sorted(boards_by_depth[depth])
        if len(candidates) < PUZZLES_PER_DEPTH:
            raise ValueError(
                f"depth {depth}: requested {PUZZLES_PER_DEPTH} puzzles but only "
                f"{len(candidates)} remain after exclusions"
            )
        for board in rng.sample(candidates, PUZZLES_PER_DEPTH):
            records.append(make_eval_record(board, _bucket_for_depth(depth), paths[board]))

    records.sort(key=lambda record: (record["optimal_length"], record["board"]))
    validate_records(records, excluded_boards)
    return records


def validate_records(
    records: list[dict[str, Any]], excluded_boards: set[Board] | None = None
) -> None:
    """Validate counts, uniqueness, exclusions, and solution replay."""

    excluded_boards = excluded_boards or set()
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"expected {EXPECTED_RECORDS} records, got {len(records)}")

    expected_counts = Counter({depth: PUZZLES_PER_DEPTH for depth in DEPTHS})
    depth_counts = Counter(record["optimal_length"] for record in records)
    if depth_counts != expected_counts:
        raise ValueError(
            f"incorrect depth counts: expected {dict(expected_counts)}, "
            f"got {dict(depth_counts)}"
        )

    boards = [tuple(record["board"]) for record in records]
    if len(set(boards)) != len(boards):
        raise ValueError("generated records contain duplicate boards")
    if set(boards) & excluded_boards:
        raise ValueError("generated records overlap excluded boards")

    for record, board in zip(records, boards, strict=True):
        actions = record["optimal_actions"]
        if len(actions) != record["optimal_length"]:
            raise ValueError(f"stored solution length is invalid for board {board}")

        current = board
        for action in actions:
            if type(action) is not int or action not in adjacent_tiles(current):
                raise ValueError(f"stored solution has illegal action {action!r}")
            current = slide_tile(current, action)
        if current != GOAL:
            raise ValueError(f"stored solution does not solve board {board}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the fixed 62-puzzle 3x3 / 8-puzzle eval set."
    )
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        nargs="+",
        default=list(DEFAULT_EXCLUSIONS),
        help="Datasets whose boards must remain out of the evaluation set.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/eval_puzzles_62.jsonl")
    )
    parser.add_argument(
        "--parquet-output",
        type=Path,
        default=Path("data/eval_puzzles_62.parquet"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    excluded_boards = load_excluded_boards(args.exclude_jsonl)
    records = generate_eval_candidates(random.Random(args.seed), excluded_boards)
    write_eval_jsonl(records, args.output)
    write_eval_parquet(records, args.parquet_output)

    print(f"Generated {len(records)} eval puzzles")
    for depth, count in sorted(Counter(record["optimal_length"] for record in records).items()):
        print(f"Depth {depth:2d}: {count:2d}")
    print(f"Excluded {len(excluded_boards)} reserved boards")
    print(f"Wrote JSONL eval data at {args.output}")
    print(f"Wrote Parquet eval data at {args.parquet_output}")


if __name__ == "__main__":
    main()
