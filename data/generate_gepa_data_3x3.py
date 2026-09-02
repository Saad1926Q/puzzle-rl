"""Generate disjoint prompt-optimization splits for the 8-puzzle."""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from evaluation.generation import (
    enumerate_from_goal,
    make_eval_record,
    write_eval_jsonl,
    write_eval_parquet,
)
from puzzle3.board import Board

DEFAULT_REPO_ID = "saad1926q/8-puzzle"
DEFAULT_CONFIG_NAME = "gepa"
DEFAULT_SEED = 42
DEFAULT_SOURCE_REVISION = "a1ba65ae934f925019fb0492330a8b56f5cb51d2"
DEFAULT_EXCLUDE_CONFIGS = ("eval", "exhaustive")

# Each tuple is (bucket label, minimum depth, maximum depth, number of examples).
SPLIT_BANDS = {
    "train": (
        ("bucket_1", 9, 11, 12),
        ("bucket_2", 12, 16, 9),
        ("bucket_3", 17, 22, 6),
        ("bucket_4", 23, 30, 3),
    ),
    "validation": (
        ("bucket_1", 9, 11, 8),
        ("bucket_2", 12, 16, 6),
        ("bucket_3", 17, 22, 4),
        ("bucket_4", 23, 30, 2),
    ),
    "test": (
        ("bucket_1", 9, 11, 12),
        ("bucket_2", 12, 16, 12),
        ("bucket_3", 17, 22, 10),
        ("bucket_4", 23, 30, 6),
    ),
}
EXPECTED_SPLIT_SIZES = {"train": 30, "validation": 20, "test": 40}


def load_excluded_boards(
    repo_id: str,
    config_names: list[str] | tuple[str, ...],
    *,
    revision: str,
) -> set[Board]:
    """Load reserved boards from pinned Hugging Face dataset configurations."""

    boards: set[Board] = set()
    for config_name in config_names:
        dataset = load_dataset(
            repo_id,
            config_name,
            split="eval",
            revision=revision,
        )
        for row_number, record in enumerate(dataset, start=1):
            board = tuple(record["board"])
            if len(board) != 9:
                raise ValueError(
                    f"{repo_id}/{config_name}:{row_number}: board must contain 9 tiles"
                )
            boards.add(board)
    return boards


def _balanced_depth_counts(
    minimum: int, maximum: int, count: int, rng: random.Random
) -> Counter[int]:
    depths = list(range(minimum, maximum + 1))
    base, remainder = divmod(count, len(depths))
    counts = Counter({depth: base for depth in depths})
    rng.shuffle(depths)
    counts.update(depths[:remainder])
    return counts


def generate_gepa_splits(
    rng: random.Random, excluded_boards: set[Board]
) -> dict[str, list[dict[str, Any]]]:
    """Generate mutually disjoint, depth-balanced GEPA train/validation/test splits."""

    paths = enumerate_from_goal()
    available: dict[int, list[Board]] = defaultdict(list)
    for board, path in paths.items():
        if board not in excluded_boards:
            available[len(path)].append(board)
    for boards in available.values():
        rng.shuffle(boards)

    splits: dict[str, list[dict[str, Any]]] = {}
    selected: set[Board] = set()
    for split, bands in SPLIT_BANDS.items():
        records: list[dict[str, Any]] = []
        for bucket, minimum, maximum, count in bands:
            depth_counts = _balanced_depth_counts(minimum, maximum, count, rng)
            for depth in range(minimum, maximum + 1):
                needed = depth_counts[depth]
                if len(available[depth]) < needed:
                    raise ValueError(
                        f"depth {depth} needs {needed} states but only "
                        f"{len(available[depth])} remain"
                    )
                for _ in range(needed):
                    board = available[depth].pop()
                    selected.add(board)
                    records.append(make_eval_record(board, bucket, paths[board]))
        rng.shuffle(records)
        splits[split] = records

    if len(selected) != sum(EXPECTED_SPLIT_SIZES.values()):
        raise AssertionError("generated GEPA splits are not mutually disjoint")
    return splits


def write_splits(
    splits: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    """Write every GEPA split as JSONL and Parquet."""

    for split, records in splits.items():
        write_eval_jsonl(records, output_dir / f"{split}.jsonl")
        write_eval_parquet(records, output_dir / f"{split}.parquet")


def push_splits(
    splits: dict[str, list[dict[str, Any]]],
    *,
    repo_id: str,
    config_name: str,
) -> None:
    """Publish all splits as one Hugging Face dataset configuration."""

    dataset = DatasetDict(
        {split: Dataset.from_list(records) for split, records in splits.items()}
    )
    dataset.push_to_hub(
        repo_id,
        config_name=config_name,
        commit_message="Add GEPA prompt-optimization splits",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate disjoint GEPA prompt-optimization puzzle splits."
    )
    parser.add_argument("--source-repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--exclude-config",
        action="append",
        default=None,
        help="Source dataset configuration to exclude; may be repeated",
    )
    parser.add_argument(
        "--source-revision",
        default=DEFAULT_SOURCE_REVISION,
        help="Pinned Hugging Face dataset revision used for exclusions",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/gepa"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    args = parser.parse_args()

    exclude_configs = args.exclude_config or list(DEFAULT_EXCLUDE_CONFIGS)
    excluded = load_excluded_boards(
        args.source_repo_id,
        exclude_configs,
        revision=args.source_revision,
    )
    splits = generate_gepa_splits(random.Random(args.seed), excluded)
    write_splits(splits, args.output_dir)

    for split, records in splits.items():
        depth_counts = Counter(record["optimal_length"] for record in records)
        print(f"{split}: {len(records)} puzzles; depths={dict(sorted(depth_counts.items()))}")
    print(f"Excluded {len(excluded)} existing evaluation boards")
    print(f"Wrote GEPA splits to {args.output_dir}")

    if args.push_to_hub:
        push_splits(splits, repo_id=args.repo_id, config_name=args.config_name)
        print(f"Pushed config {args.config_name!r} to {args.repo_id}")


if __name__ == "__main__":
    main()
