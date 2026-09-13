"""Model-facing board serializations used by representation ablations."""

from __future__ import annotations

from typing import Literal

from puzzle3.board import Board, WIDTH

BoardRepresentation = Literal["grid", "rows", "markdown", "coordinates"]
BOARD_REPRESENTATIONS: tuple[BoardRepresentation, ...] = (
    "grid",
    "rows",
    "markdown",
    "coordinates",
)


def render_board(
    board: Board,
    representation: BoardRepresentation = "grid",
) -> str:
    """Render a board using a model-facing representation."""
    rows = [board[row * WIDTH : (row + 1) * WIDTH] for row in range(WIDTH)]
    if representation == "grid":
        return "\n".join(" ".join(str(tile) for tile in row) for row in rows)
    if representation == "rows":
        return "\n".join(
            f"Row {index}: " + " ".join(str(tile) for tile in row)
            for index, row in enumerate(rows, start=1)
        )
    if representation == "markdown":
        lines = [
            "|    | C1 | C2 | C3 |",
            "|----|----|----|----|",
        ]
        lines.extend(
            f"| R{index} | " + " | ".join(str(tile) for tile in row) + " |"
            for index, row in enumerate(rows, start=1)
        )
        return "\n".join(lines)
    if representation == "coordinates":
        return "\n".join(
            ", ".join(
                f"R{row_index}C{column_index}={tile}"
                for column_index, tile in enumerate(row, start=1)
            )
            for row_index, row in enumerate(rows, start=1)
        )
    raise ValueError(f"unsupported board representation: {representation}")
