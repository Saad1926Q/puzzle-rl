"""Generate the fixed exhaustive 8-puzzle evaluation dataset."""

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

DEPTH_COUNTS = {
    3: 4,
    4: 8,
    5: 10,
    6: 20,
    **dict.fromkeys(range(7, 11), 25),
    **dict.fromkeys(range(11, 16), 10),
    # Depth 31 has only two states and the original eval excludes one.
    # Reassign the four unavailable depth-31 slots to depth 30.
    **dict.fromkeys(range(16, 30), 5),
    30: 9,
    31: 1,
}
EXPECTED_REACHABLE_STATES = 181_440
EXPECTED_MAX_DEPTH = 31
EXPECTED_RECORDS = 272


def _bucket_for_depth(depth: int) -> str:
    if depth <= 5:
        return "easy"
    if depth <= 15:
        return "medium"
    return "hard"


def load_excluded_boards(path: Path) -> set[Board]:
    """Load boards that must not appear in the new evaluation dataset."""

    if not path.is_file():
        raise FileNotFoundError(f"exclusion dataset does not exist: {path}")

    excluded: set[Board] = set()
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


def generate_exhaustive_eval_candidates(
    rng: random.Random,
    excluded_boards: set[Board],
) -> list[dict[str, Any]]:
    """Sample the configured number of unseen boards at every target depth."""

    paths = enumerate_from_goal()
    if len(paths) != EXPECTED_REACHABLE_STATES:
        raise RuntimeError(
            f"expected {EXPECTED_REACHABLE_STATES} reachable states, got {len(paths)}"
        )
    if max(map(len, paths.values())) != EXPECTED_MAX_DEPTH:
        raise RuntimeError(f"expected maximum depth {EXPECTED_MAX_DEPTH}")

    boards_by_depth: dict[int, list[Board]] = defaultdict(list)
    for board, path in paths.items():
        if board not in excluded_boards and len(path) in DEPTH_COUNTS:
            boards_by_depth[len(path)].append(board)

    records: list[dict[str, Any]] = []
    for depth, count in DEPTH_COUNTS.items():
        candidates = sorted(boards_by_depth[depth])
        if len(candidates) < count:
            raise ValueError(
                f"depth {depth}: requested {count} puzzles but only "
                f"{len(candidates)} remain after exclusions"
            )
        for board in rng.sample(candidates, count):
            records.append(
                make_eval_record(board, _bucket_for_depth(depth), paths[board])
            )

    records.sort(key=lambda record: (record["optimal_length"], record["board"]))
    validate_records(records, excluded_boards)
    return records


def validate_records(
    records: list[dict[str, Any]], excluded_boards: set[Board]
) -> None:
    """Validate counts, uniqueness, optimal lengths, and action replay."""

    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"expected {EXPECTED_RECORDS} records, got {len(records)}")

    depth_counts = Counter(record["optimal_length"] for record in records)
    if depth_counts != Counter(DEPTH_COUNTS):
        raise ValueError(
            f"incorrect depth counts: expected {DEPTH_COUNTS}, got {dict(depth_counts)}"
        )

    boards = [tuple(record["board"]) for record in records]
    if len(set(boards)) != len(boards):
        raise ValueError("generated records contain duplicate boards")
    if set(boards) & excluded_boards:
        raise ValueError("generated records overlap the exclusion dataset")

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
        description="Generate the fixed 272-puzzle exhaustive eval set."
    )
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        default=Path("data/eval_puzzles_3x3_45.jsonl"),
        help="Existing evaluation JSONL whose boards must be excluded.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval_puzzles_3x3_exhaustive_272.jsonl"),
    )
    parser.add_argument(
        "--parquet-output",
        type=Path,
        default=Path("data/eval_puzzles_3x3_exhaustive_272.parquet"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    excluded_boards = load_excluded_boards(args.exclude_jsonl)
    records = generate_exhaustive_eval_candidates(
        random.Random(args.seed), excluded_boards
    )
    write_eval_jsonl(records, args.output)
    write_eval_parquet(records, args.parquet_output)

    print(f"Generated {len(records)} exhaustive 3x3 eval puzzles")
    for depth, count in DEPTH_COUNTS.items():
        print(f"Depth {depth:2d}: {count:2d}")
    print(f"Excluded {len(excluded_boards)} existing evaluation boards")
    print(f"Wrote JSONL eval data at {args.output}")
    print(f"Wrote Parquet eval data at {args.parquet_output}")


if __name__ == "__main__":
    main()
