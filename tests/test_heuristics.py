import pytest

from puzzle.heuristics import (
    col_linear_conflicts,
    manhattan,
    manhattan_linear_conflict,
    row_linear_conflicts,
)


@pytest.mark.parametrize(
    "board,expected",
    [
        ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0), 0),
        ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 0), 2),
    ],
)
def test_manhattan(board: tuple[int, ...], expected: int):
    assert manhattan(board) == expected


@pytest.mark.parametrize(
    "board,expected",
    [
        ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0), 0),
        ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 14, 13, 0), 6),
    ],
)
def test_row_linear_conflicts(board: tuple[int, ...], expected: int):
    assert row_linear_conflicts(board) == expected


@pytest.mark.parametrize(
    "board,expected",
    [
        ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0), 0),
        ((13, 2, 3, 4, 9, 6, 7, 8, 5, 10, 11, 12, 1, 14, 15, 0), 12),
    ],
)
def test_col_linear_conflicts(board: tuple[int, ...], expected: int):
    assert col_linear_conflicts(board) == expected


def test_manhattan_linear_conflict():
    board = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 14, 13, 0)

    assert manhattan(board) == 4
    assert row_linear_conflicts(board) == 6
    assert col_linear_conflicts(board) == 0
    assert manhattan_linear_conflict(board) == 10
