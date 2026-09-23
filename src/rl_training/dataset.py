"""Dataset loading and normalization for puzzle RL episodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from puzzle3.episode import DEFAULT_MAX_TURNS, PuzzleEpisode

DEFAULT_PROMPT = "Solve this 8-puzzle one move at a time using the slide_tile tool."


def _normalize_row(
    row: dict[str, Any],
    *,
    max_turns: int,
) -> dict[str, Any]:
    """Validate one puzzle row and add the conversational prompt TRL expects."""

    try:
        board = tuple(int(value) for value in row["board"])
        optimal_length = row["optimal_length"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("RL rows require board and optimal_length fields") from error
    PuzzleEpisode(
        initial_board=board,
        optimal_length=optimal_length,
        max_turns=max_turns,
    )
    return {
        "board": list(board),
        "optimal_length": optimal_length,
        "max_turns": max_turns,
        "prompt": [{"role": "user", "content": DEFAULT_PROMPT}],
    }


def load_puzzle_dataset(
    source: str,
    *,
    config: str | None = None,
    split: str = "train",
    max_turns: int = DEFAULT_MAX_TURNS,
) -> Dataset:
    """Load local JSON/JSONL or a Hub dataset and normalize its puzzle rows."""

    path = Path(source)
    if path.is_file():
        dataset = load_dataset(
            "json",
            data_files={split: str(path)},
            split=split,
        )
    else:
        dataset = load_dataset(source, config, split=split)
    if len(dataset) == 0:
        raise ValueError("RL dataset is empty")
    return dataset.map(
        _normalize_row,
        fn_kwargs={"max_turns": max_turns},
        desc="Validating puzzle RL rows",
    )
