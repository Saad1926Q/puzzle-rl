from typing import List, Tuple

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
WIDTH = 4


def index_to_rc(idx: int) -> Tuple[int, int]:
    """
    Convert index of flat board to (row,col) for 4 x 4 grid.
    """

    row = idx // WIDTH

    col = idx % WIDTH

    return (row, col)


def rc_to_index(r: int, c: int) -> int:
    """
    Convert (row,col) to flat index for 4 x 4 grid. Inverse of index_to_rc.
    """

    return r * WIDTH + c


def get_blank_idx(board: Tuple[int, ...]) -> int:
    """
    Get the flat index of the blank (0) tile in the given board.
    """

    return board.index(0)


def get_goal_rc(tile_value: int) -> Tuple[int, int]:
    """
    Get the (row,col) that the given tile value belongs at in GOAL.
    """

    idx = GOAL.index(tile_value)

    return index_to_rc(idx)


def legal_moves(board: Tuple[int, ...]) -> List[str]:
    """
    Get the list of currently legal move commands for the given board.

    Follows conventions from TextArena's implementation: a command names
    what the tile does, not the blank, e.g. "up" means the tile below the
    blank slides up into it. A command is legal only if the blank has a
    neighbor in the opposite direction to pull a tile from:
      - "up":    blank has a row below it   (blank_r < 3)
      - "down":  blank has a row above it   (blank_r > 0)
      - "left":  blank has a column to its right (blank_c < 3)
      - "right": blank has a column to its left  (blank_c > 0)
    """

    blank_idx = get_blank_idx(board)

    blank_r, blank_c = index_to_rc(blank_idx)

    moves = []

    if blank_r < 3:
        moves.append("up")

    if blank_r > 0:
        moves.append("down")

    if blank_c < 3:
        moves.append("left")

    if blank_c > 0:
        moves.append("right")

    return moves
