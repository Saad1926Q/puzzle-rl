GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
WIDTH = 3
MOVE_DELTA = {"up": (1, 0), "down": (-1, 0), "left": (0, 1), "right": (0, -1)}


def index_to_rc(idx: int) -> tuple[int, int]:
    """Convert index of flat board to (row, col) for a 3x3 grid."""

    return (idx // WIDTH, idx % WIDTH)


def rc_to_index(r: int, c: int) -> int:
    """Convert (row, col) to flat index for a 3x3 grid."""

    return r * WIDTH + c


def _get_opposite(move: str):
    """Return the move that immediately undoes the given move."""

    if move == "up":
        return "down"
    if move == "down":
        return "up"
    if move == "left":
        return "right"
    return "left"


def _build_goal_pos() -> dict[int, tuple[int, int]]:
    """Build a lookup from tile value to its goal (row, col) position."""

    return {tile: index_to_rc(idx) for idx, tile in enumerate(GOAL)}


def _build_legal_move_table() -> tuple[tuple[str, ...], ...]:
    """Precompute legal move names for each possible blank position."""

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
    """Get the index of the blank (0) tile."""

    return board.index(0)


def get_goal_rc(tile_value: int) -> tuple[int, int]:
    """Get the (row, col) that the given tile value belongs at in GOAL."""

    return GOAL_POS[tile_value]


def legal_moves(board: tuple[int, ...]) -> list[str]:
    """
    Get legal move commands for the board.

    Tile-moving convention: a command names what the numbered tile does.
    For example, "left" means the tile to the right of the blank slides left into it.
    """

    return list(LEGAL_MOVE_TABLE[get_blank_idx(board)])


def get_non_reversing_moves(board: tuple[int, ...], last_move: str | None) -> list[str]:
    """Get legal moves while excluding the immediate reversal of last_move."""

    moves = legal_moves(board)
    if last_move is None:
        return moves
    return [move for move in moves if move != _get_opposite(last_move)]


def apply_move(board: tuple[int, ...], move: str) -> tuple[int, ...]:
    """Apply a legal move and return the resulting board."""

    new_board = list(board)
    blank_idx = get_blank_idx(board)
    blank_r, blank_c = index_to_rc(blank_idx)
    dr, dc = MOVE_DELTA[move]
    target_idx = rc_to_index(blank_r + dr, blank_c + dc)
    new_board[blank_idx], new_board[target_idx] = new_board[target_idx], new_board[blank_idx]
    return tuple(new_board)


def is_solved(board: tuple[int, ...]) -> bool:
    """Check whether the board is in the solved state."""

    return board == GOAL
