from puzzle.board import _get_opposite, apply_move, is_solved, legal_moves
from puzzle.heuristics import manhattan_linear_conflict


def _search(
    board: tuple[int, ...], path: list[str], g: int, threshold: int | float, last_move
) -> tuple[bool, float]:
    """
    Run one depth-first IDA* search iteration up to the current threshold.
    """

    h = manhattan_linear_conflict(board)

    f = g + h

    if f > threshold:
        return False, f

    if is_solved(board=board):
        return True, f

    min_exceeded = float("inf")

    for move in legal_moves(board):
        if last_move is not None and move == _get_opposite(last_move):
            continue

        next_board = apply_move(board=board, move=move)

        path.append(move)

        is_found, result = _search(next_board, path, g + 1, threshold, move)

        if is_found:
            return True, result

        path.pop()

        min_exceeded = min(min_exceeded, result)

    return False, min_exceeded


def solve(board: tuple[int, ...]) -> list[str]:
    """
    Solve the puzzle with IDA* and return the sequence of move commands.
    """

    threshold = manhattan_linear_conflict(board)

    path = []

    while True:
        is_found, value = _search(
            board=board, path=path, g=0, threshold=threshold, last_move=None
        )

        if is_found:
            return path

        threshold = value
