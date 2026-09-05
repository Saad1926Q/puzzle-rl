"""Generate fresh, depth-balanced 8-puzzle boards for teacher rollouts.

This source data deliberately stores no optimal actions.  Exact depths are kept
only for sampling and later analysis; a rollout model receives the board alone.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.constants import ACTION_INTERFACE
from evaluation.generation import enumerate_from_goal, write_eval_jsonl, write_eval_parquet
from puzzle3.board import Board, GOAL

DEPTHS = tuple(range(12, 17))
TARGET_PER_DEPTH = 300
EXPECTED_RECORDS = len(DEPTHS) * TARGET_PER_DEPTH
DEFAULT_EXCLUSIONS = (
    Path("data/eval_puzzles_3x3_45.jsonl"),
    Path("data/eval_puzzles_3x3_exhaustive_272.jsonl"),
)


def load_excluded_boards(paths: list[Path]) -> set[Board]:
    """Load valid boards from every reserved JSONL dataset."""

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


def generate_sft_source(
    rng: random.Random,
    excluded_boards: set[Board],
) -> list[dict[str, Any]]:
    """Sample fresh boards uniformly within each requested exact-depth band."""

    paths = enumerate_from_goal()
    candidates_by_depth: dict[int, list[Board]] = defaultdict(list)
    for board, path_from_goal in paths.items():
        depth = len(path_from_goal)
        if depth in DEPTHS and board not in excluded_boards:
            candidates_by_depth[depth].append(board)

    records: list[dict[str, Any]] = []
    for depth in DEPTHS:
        candidates = sorted(candidates_by_depth[depth])
        if len(candidates) < TARGET_PER_DEPTH:
            raise ValueError(
                f"depth {depth}: requested {TARGET_PER_DEPTH} puzzles but only "
                f"{len(candidates)} remain after exclusions"
            )
        for board in rng.sample(candidates, TARGET_PER_DEPTH):
            records.append(
                {
                    "board": list(board),
                    "optimal_length": depth,
                    "action_interface": ACTION_INTERFACE,
                }
            )

    records.sort(key=lambda record: (record["optimal_length"], record["board"]))
    for index, record in enumerate(records, start=1):
        record["id"] = f"sft-source-{index:04d}"
    validate_records(records, excluded_boards)
    return records


def validate_records(
    records: list[dict[str, Any]], excluded_boards: set[Board]
) -> None:
    """Validate source-only schema, depth balance, uniqueness, and reservation."""

    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"expected {EXPECTED_RECORDS} records, got {len(records)}")
    if Counter(record["optimal_length"] for record in records) != Counter(
        {depth: TARGET_PER_DEPTH for depth in DEPTHS}
    ):
        raise ValueError("source records have incorrect depth counts")
    boards = [tuple(record["board"]) for record in records]
    if len(set(boards)) != len(boards):
        raise ValueError("source records contain duplicate boards")
    if set(boards) & excluded_boards:
        raise ValueError("source records overlap reserved boards")
    if any("optimal_actions" in record for record in records):
        raise ValueError("source records must not expose optimal actions")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 1,500 fresh 8-puzzle teacher-rollout boards at depths 12-16."
    )
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        nargs="+",
        default=list(DEFAULT_EXCLUSIONS),
        help="Reserved JSONL datasets whose boards must be excluded.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sft_source_3x3_depths_12_16_1500.jsonl"),
    )
    parser.add_argument(
        "--parquet-output",
        type=Path,
        default=Path("data/sft_source_3x3_depths_12_16_1500.parquet"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    excluded_boards = load_excluded_boards(args.exclude_jsonl)
    records = generate_sft_source(random.Random(args.seed), excluded_boards)
    write_eval_jsonl(records, args.output)
    write_eval_parquet(records, args.parquet_output)

    print(f"Generated {len(records)} fresh SFT source puzzles")
    for depth in DEPTHS:
        print(f"Depth {depth:2d}: {TARGET_PER_DEPTH}")
    print(f"Excluded {len(excluded_boards)} reserved boards")
    print(f"Wrote JSONL source data at {args.output}")
    print(f"Wrote Parquet source data at {args.parquet_output}")


if __name__ == "__main__":
    main()
