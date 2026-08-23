"""Dataset loading and validation for the 8-puzzle evaluation set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from evaluation.constants import VALID_MOVE_SET
from puzzle3.board import GOAL

DEFAULT_DATASET = "saad1926q/8-puzzle"
DEFAULT_CONFIG = "eval"
DEFAULT_SPLIT = "eval"


class DatasetError(ValueError):
    """Raised when a dataset row cannot be used as an 8-puzzle task."""


@dataclass(frozen=True)
class PuzzleExample:
    """One authoritative puzzle task from the evaluation dataset."""

    example_id: str
    board: tuple[int, ...]
    optimal_moves: tuple[str, ...]
    optimal_length: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "board": list(self.board),
            "optimal_moves": list(self.optimal_moves),
            "optimal_length": self.optimal_length,
            **self.metadata,
        }


def _make_example(row: dict[str, Any], example_id: str) -> PuzzleExample:
    try:
        board = tuple(int(value) for value in row["board"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetError(f"{example_id}: board must contain nine integers") from exc

    if len(board) != 9 or set(board) != set(GOAL):
        raise DatasetError(f"{example_id}: board must be a permutation of 0..8")

    try:
        optimal_moves = tuple(
            str(move).strip().lower() for move in row["optimal_moves"]
        )
        optimal_length = int(row["optimal_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetError(
            f"{example_id}: optimal_moves and optimal_length are required"
        ) from exc

    if any(move not in VALID_MOVE_SET for move in optimal_moves):
        raise DatasetError(f"{example_id}: optimal_moves contains an unknown move")
    if optimal_length != len(optimal_moves) or optimal_length < 0:
        raise DatasetError(f"{example_id}: optimal_length does not match optimal_moves")

    metadata = {
        key: value
        for key, value in row.items()
        if key not in {"board", "optimal_moves", "optimal_length"}
    }
    return PuzzleExample(example_id, board, optimal_moves, optimal_length, metadata)


def _load_rows(
    dataset: str,
    *,
    config: str,
    split: str,
) -> Any:
    """Load rows from a local JSONL subset or from Hugging Face."""

    local_path = Path(dataset)
    if not local_path.is_file():
        return load_dataset(dataset, name=config, split=split)
    if local_path.suffix == ".parquet":
        return load_dataset(
            "parquet",
            data_files={split: str(local_path)},
            split=split,
        )
    if local_path.suffix != ".jsonl":
        raise DatasetError(
            f"local dataset must be a .jsonl or .parquet file: {dataset}"
        )

    rows = []
    with local_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{dataset}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{dataset}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def load_examples(
    *,
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    limit: int | None = None,
    offset: int = 0,
) -> list[PuzzleExample]:
    """Load a deterministic slice from Hugging Face or a local JSONL/Parquet file."""

    if (limit is not None and limit <= 0) or offset < 0:
        raise ValueError(
            "limit must be positive when provided and offset must be non-negative"
        )

    rows = _load_rows(dataset, config=config, split=split)
    end = len(rows) if limit is None else min(offset + limit, len(rows))
    if limit is not None and end - offset != limit:
        raise DatasetError(
            f"{dataset}/{config}/{split}: expected {limit} rows from offset {offset}, "
            f"found {max(0, end - offset)}"
        )

    return [
        _make_example(rows[index], f"{dataset}:{config}:{split}:{index}")
        for index in range(offset, end)
    ]
