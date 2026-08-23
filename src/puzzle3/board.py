type Board = tuple[int, ...]
type TileAction = int

GOAL: Board = (1, 2, 3, 4, 5, 6, 7, 8, 0)
WIDTH = 3


def index_to_rc(idx: int) -> tuple[int, int]:
    """Convert index of flat board to (row, col) for a 3x3 grid."""

    return (idx // WIDTH, idx % WIDTH)


def rc_to_index(r: int, c: int) -> int:
    """Convert (row, col) to index for a 3x3 grid."""

    return r * WIDTH + c


def _build_goal_pos() -> dict[int, tuple[int, int]]:
    """Build a lookup from tile value to its goal (row, col) position."""

    return {tile: index_to_rc(idx) for idx, tile in enumerate(GOAL)}


GOAL_POS = _build_goal_pos()

# Board connectivity for each possible blank index, in stable row-major order.
_NEIGHBOR_INDICES: tuple[tuple[int, ...], ...] = (
    (1, 3),
    (0, 2, 4),
    (1, 5),
    (0, 4, 6),
    (1, 3, 5, 7),
    (2, 4, 8),
    (3, 7),
    (4, 6, 8),
    (5, 7),
)


def get_blank_idx(board: Board) -> int:
    """Get the index of the blank (0) tile."""

    return board.index(0)


def get_goal_rc(tile_value: int) -> tuple[int, int]:
    """Get the (row, col) that the given tile value belongs at in GOAL."""

    return GOAL_POS[tile_value]


def adjacent_tiles(board: Board) -> tuple[int, ...]:
    """Return tile IDs that can currently slide into the blank."""

    blank_idx = get_blank_idx(board)
    return tuple(board[index] for index in _NEIGHBOR_INDICES[blank_idx])


def slide_tile(board: Board, tile: TileAction) -> Board:
    """Slide an adjacent numbered tile into the blank."""

    if type(tile) is not int or not 1 <= tile <= 8:
        raise ValueError("tile must be an integer from 1 through 8")

    try:
        blank_idx = get_blank_idx(board)
        tile_idx = board.index(tile)
    except ValueError as exc:
        raise ValueError(f"tile {tile} and blank must be present on the board") from exc

    blank_r, blank_c = index_to_rc(blank_idx)
    tile_r, tile_c = index_to_rc(tile_idx)
    if abs(blank_r - tile_r) + abs(blank_c - tile_c) != 1:
        raise ValueError(f"tile {tile} is not adjacent to the blank")

    new_board = list(board)
    new_board[blank_idx], new_board[tile_idx] = (
        new_board[tile_idx],
        new_board[blank_idx],
    )
    return tuple(new_board)


def is_solved(board: Board) -> bool:
    """Check whether the board is in the solved state."""

    return board == GOAL
