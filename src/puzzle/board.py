GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
WIDTH = 4
MOVE_DELTA = {"up": (1, 0), "down": (-1, 0), "left": (0, 1), "right": (0, -1)}


def index_to_rc(idx: int) -> tuple[int, int]:
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


def _build_goal_pos() -> dict[int, tuple[int, int]]:
    return {tile: index_to_rc(idx) for idx, tile in enumerate(GOAL)}


def _build_legal_move_table() -> tuple[tuple[str, ...], ...]:
    table = []

    for blank_idx in range(len(GOAL)):
        blank_r, blank_c = index_to_rc(blank_idx)
        moves = []

        if blank_r < WIDTH - 1:
            moves.append("up")

        if blank_r > 0:
            moves.append("down")

        if blank_c < WIDTH - 1:
            moves.append("left")

        if blank_c > 0:
            moves.append("right")

        table.append(tuple(moves))

    return tuple(table)


GOAL_POS = _build_goal_pos()
LEGAL_MOVE_TABLE = _build_legal_move_table()


def get_blank_idx(board: tuple[int, ...]) -> int:
    """
    Get the index of the blank(0) tile in the given board.
    """

    return board.index(0)


def get_goal_rc(tile_value: int) -> tuple[int, int]:
    """
    Get the (row,col) that the given tile value belongs at in GOAL.
    """

    return GOAL_POS[tile_value]


def legal_moves(board: tuple[int, ...]) -> list[str]:
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

    return list(LEGAL_MOVE_TABLE[blank_idx])


def apply_move(board: tuple[int, ...], move: str) -> tuple[int, ...]:
    """
    Apply a move to the board and return the resulting board.

    Assumes a legal move is passed, hence no validation.
    """

    new_board = list(board)

    blank_idx = get_blank_idx(board)

    blank_r, blank_c = index_to_rc(blank_idx)

    dr, dc = MOVE_DELTA[move]

    target_r, target_c = blank_r + dr, blank_c + dc

    target_idx = rc_to_index(target_r, target_c)

    new_board[blank_idx], new_board[target_idx] = (
        new_board[target_idx],
        new_board[blank_idx],
    )

    return tuple(new_board)


def is_solved(board: tuple[int, ...]) -> bool:
    """
    Check whether the board is in the solved state.
    """

    return board == GOAL
