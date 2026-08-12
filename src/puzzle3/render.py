from puzzle3.board import WIDTH


def render(board: tuple[int, ...]) -> str:
    """Render the board as a 3x3 text grid, with the blank tile shown as '_'."""

    rows = []
    for r in range(WIDTH):
        row_tiles = board[r * WIDTH : (r + 1) * WIDTH]
        rows.append(" ".join("_" if tile == 0 else str(tile) for tile in row_tiles))
    return "\n".join(rows)
