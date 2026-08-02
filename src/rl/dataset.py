"""Dataset rows for TRL training."""

from __future__ import annotations

from typing import Any, Sequence

from datasets import Dataset, load_dataset

from env.protocol import build_board_prompt

DATASET_NAME = "saad1926q/15-puzzle"
SUBSET = "rl"
SPLIT = "rl"
KEEP_FIELDS = ("board", "optimal_length", "optimal_moves", "bucket", "scramble_depth", "id")


def load_rl_dataset(
    dataset_name: str = DATASET_NAME,
    subset: str = SUBSET,
    split: str = SPLIT,
) -> Dataset:
    """Load the puzzle RL split."""

    return load_dataset(dataset_name, subset, split=split)


def rows_from_dataset(dataset: Dataset) -> list[dict[str, Any]]:
    """Convert a HuggingFace dataset to plain dicts."""

    return [dict(row) for row in dataset]


def build_trl_row(row: dict[str, Any]) -> dict[str, Any]:
    """Wrap one puzzle row for the rollout function."""

    board = tuple(row["board"])
    prompt = {
        "messages": [{"role": "user", "content": build_board_prompt(board)}],
        "initial_board": list(board),
        "optimal_length": int(row["optimal_length"]),
        "optimal_moves": list(row.get("optimal_moves") or []),
        "bucket": row["bucket"],
        "scramble_depth": int(row.get("scramble_depth", 0)),
        "id": row.get("id"),
    }

    trl_row: dict[str, Any] = {"prompt": prompt}
    for key in KEEP_FIELDS:
        trl_row[key] = row[key]
    return trl_row


def build_train_dataset_from_rows(rows: Sequence[dict[str, Any]]) -> Dataset:
    """Build the full training dataset."""

    return Dataset.from_list([build_trl_row(row) for row in rows])
