from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

from puzzle3.board import GOAL


_SPEC = importlib.util.spec_from_file_location(
    "create_sft_validation_3x3",
    Path(__file__).parents[1] / "data" / "create_sft_validation_3x3.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_validation_puzzles_cover_depths_without_solution_labels() -> None:
    records = _MODULE.generate_validation_puzzles(
        _MODULE.random.Random(42),
        excluded_boards=set(),
    )

    assert len(records) == 10
    assert Counter(record["optimal_length"] for record in records) == {
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 2,
    }
    assert len({tuple(record["board"]) for record in records}) == 10
    assert all(set(record["board"]) == set(GOAL) for record in records)
    assert all("optimal_actions" not in record for record in records)
