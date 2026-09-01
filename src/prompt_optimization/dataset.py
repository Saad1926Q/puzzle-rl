"""Load the disjoint datasets used for GEPA prompt optimization."""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.dataset import PuzzleExample, load_examples

DEFAULT_GEPA_CONFIG = "gepa"
GEPA_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class GEPASplits:
    """Disjoint task splits with distinct optimization roles."""

    train: list[PuzzleExample]
    validation: list[PuzzleExample]
    test: list[PuzzleExample]


def load_gepa_splits(
    *,
    dataset: str = "saad1926q/8-puzzle",
    config: str = DEFAULT_GEPA_CONFIG,
) -> GEPASplits:
    """Load all GEPA splits without exposing them to the task model as labels."""

    splits = {
        split: load_examples(dataset=dataset, config=config, split=split)
        for split in GEPA_SPLITS
    }
    return GEPASplits(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
    )
