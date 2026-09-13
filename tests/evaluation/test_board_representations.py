from __future__ import annotations

import pytest

from evaluation.board_representations import render_board


BOARD = (1, 0, 3, 8, 2, 5, 4, 7, 6)


@pytest.mark.parametrize(
    ("representation", "expected"),
    [
        ("grid", "1 0 3\n8 2 5\n4 7 6"),
        (
            "rows",
            "Row 1: 1 0 3\nRow 2: 8 2 5\nRow 3: 4 7 6",
        ),
        (
            "markdown",
            "|    | C1 | C2 | C3 |\n"
            "|----|----|----|----|\n"
            "| R1 | 1 | 0 | 3 |\n"
            "| R2 | 8 | 2 | 5 |\n"
            "| R3 | 4 | 7 | 6 |",
        ),
        (
            "coordinates",
            "R1C1=1, R1C2=0, R1C3=3\n"
            "R2C1=8, R2C2=2, R2C3=5\n"
            "R3C1=4, R3C2=7, R3C3=6",
        ),
    ],
)
def test_render_board_representations(representation: str, expected: str) -> None:
    assert render_board(BOARD, representation) == expected


def test_render_board_defaults_to_existing_grid() -> None:
    assert render_board(BOARD) == "1 0 3\n8 2 5\n4 7 6"


def test_render_board_rejects_unknown_representation() -> None:
    with pytest.raises(ValueError, match="unsupported board representation"):
        render_board(BOARD, "unknown")  # type: ignore[arg-type]
