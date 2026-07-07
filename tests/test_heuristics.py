import pytest

from puzzle.heuristics import manhattan


@pytest.mark.parametrize(
    "board,expected",
    [
        ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0), 0),
        ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 0), 2),
    ],
)
def test_manhattan(board: tuple[int, ...], expected: int):
    assert manhattan(board) == expected
