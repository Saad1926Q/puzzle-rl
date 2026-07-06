from typing import Tuple

import pytest

from puzzle.board import get_blank_idx, get_goal_rc, index_to_rc, legal_moves, rc_to_index


@pytest.mark.parametrize(
    "idx,expected", [(0, (0, 0)), (15, (3, 3)), (4, (1, 0)), (7, (1, 3))]
)
def test_index_to_rc(idx: int, expected: Tuple[int, int]):
    assert index_to_rc(idx) == expected


@pytest.mark.parametrize(
    "rc,expected", [((0, 0), 0), ((3, 3), 15), ((1, 0), 4), ((1, 3), 7)]
)
def test_rc_to_index(rc: Tuple[int, int], expected: int):
    assert rc_to_index(*rc) == expected


@pytest.mark.parametrize(
    "board,expected",
    [
        ((5, 3, 0, 4, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 6), 2),
        ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0), 15),
        ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 0),
    ],
)
def test_get_blank_idx(board: Tuple[int, ...], expected: int):
    assert get_blank_idx(board) == expected


@pytest.mark.parametrize(
    "tile_value,expected", [(1, (0, 0)), (15, (3, 2)), (5, (1, 0)), (8, (1, 3))]
)
def test_get_goal_rc(tile_value: int, expected: Tuple[int, int]):
    assert get_goal_rc(tile_value) == expected


@pytest.mark.parametrize(
    "board,expected",
    [
        # blank at top-left corner (idx 0)
        ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), ["up", "left"]),
        # blank at top-right corner (idx 3)
        ((1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), ["up", "right"]),
    ],
)
def test_legal_moves(board: Tuple[int, ...], expected: list):
    assert legal_moves(board) == expected
