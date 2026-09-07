"""Build SFT training decisions and puzzle-level validation records."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from puzzle3.board import adjacent_tiles
from sft_generation.formatting import build_history_window_sft_records
from sft_generation.storage import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse dataset-building options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--validation-puzzles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-turns", type=int, default=4)
    return parser.parse_args()


def puzzle_validation_record(puzzle: dict[str, Any]) -> dict[str, Any]:
    """Represent one starting puzzle in the shared SFT split schema."""
    board = tuple(puzzle["board"])
    source_id = str(puzzle["id"])
    return {
        "prompt": [],
        "completion": [],
        "metadata": {
            "record_type": "puzzle",
            "source_id": source_id,
            "rollout_id": 0,
            "target_turn": 0,
            "history_turns": 0,
            "initial_board": list(board),
            "initial_depth": puzzle["optimal_length"],
            "optimal_actions": list(puzzle["optimal_actions"]),
            "board": list(board),
            "legal_tiles": list(adjacent_tiles(board)),
            "tile": 0,
            "next_board": list(board),
            "teacher_model": "",
        },
        "tools": [],
    }


def main() -> None:
    """Build all decision rows for train and fresh puzzles for validation."""
    args = parse_args()
    if args.history_turns < 0:
        raise ValueError("history-turns must be non-negative")

    trajectories = sorted(
        list(read_jsonl(args.trajectories)),
        key=lambda item: str(item["source_id"]),
    )
    annotations_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in read_jsonl(args.annotations):
        annotations_by_source[str(annotation["source_id"])].append(annotation)

    train_records: list[dict[str, Any]] = []
    for trajectory in trajectories:
        source_id = str(trajectory["source_id"])
        train_records.extend(
            build_history_window_sft_records(
                trajectory,
                annotations_by_source.get(source_id, []),
                history_turns=args.history_turns,
            )
        )

    validation_puzzles = sorted(
        list(read_jsonl(args.validation_puzzles)),
        key=lambda item: str(item["id"]),
    )
    validation_records = [
        puzzle_validation_record(puzzle) for puzzle in validation_puzzles
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    metadata_path = args.output_dir / "metadata.json"
    write_jsonl(train_path, train_records, sort_keys=False)
    write_jsonl(validation_path, validation_records, sort_keys=False)
    write_json(
        metadata_path,
        {
            "trajectories": str(args.trajectories),
            "annotations": str(args.annotations),
            "validation_puzzles": str(args.validation_puzzles),
            "history_turns": args.history_turns,
            "trajectory_count": len(trajectories),
            "train_record_count": len(train_records),
            "validation_puzzle_count": len(validation_records),
            "train_source_ids": [
                str(trajectory["source_id"]) for trajectory in trajectories
            ],
            "validation_source_ids": [
                str(puzzle["id"]) for puzzle in validation_puzzles
            ],
        },
    )
    print(
        f"trajectories={len(trajectories)} "
        f"train_records={len(train_records)} "
        f"validation_puzzles={len(validation_records)}"
    )
    print(f"train saved at {train_path}")
    print(f"validation saved at {validation_path}")
    print(f"metadata saved at {metadata_path}")


if __name__ == "__main__":
    main()
