from puzzle3.board import Board, WIDTH


def render(board: Board) -> str:
    """Render the board as a 3x3 text grid, including 0 for the blank tile."""

    rows = []
    for r in range(WIDTH):
        row_tiles = board[r * WIDTH : (r + 1) * WIDTH]
        rows.append(" ".join(str(tile) for tile in row_tiles))
    return "\n".join(rows)
