from puzzle.board import WIDTH


def render(board: tuple[int, ...]) -> str:
    """
    Render the board as a 4x4 text grid, with the blank tile shown as '_'.
    """

    rows = []

    for r in range(WIDTH):
        row_tiles = board[r * WIDTH : (r + 1) * WIDTH]
        row_str = " ".join("_" if tile == 0 else str(tile) for tile in row_tiles)
        rows.append(row_str)

    return "\n".join(rows)
