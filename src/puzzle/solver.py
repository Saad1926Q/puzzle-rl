from puzzle.board import apply_move, is_solved, legal_moves
from puzzle.heuristics import manhattan_linear_conflict


def _get_opposite(move: str):
    if move == "up":
        return "down"
    elif move == "down":
        return "up"
    elif move == "left":
        return "right"
    else:
        return "left"


def _search(
    board: tuple[int, ...], path: list[str], g: int, threshold: int | float, last_move
) -> tuple[bool, float]:
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
    threshold = manhattan_linear_conflict(board)

    path = []

    while True:
        is_found, value = _search(
            board=board, path=path, g=0, threshold=threshold, last_move=None
        )

        if is_found:
            return path

        threshold = value
