from collections import deque

from puzzle3.board import GOAL, _get_opposite, apply_move, is_solved, legal_moves


def solve(board: tuple[int, ...]) -> list[str]:
    """
    Solve an 8-puzzle board optimally using BFS.

    The 8-puzzle has only 181,440 solvable states, so BFS is simple and robust
    for eval-set generation and diagnostics.
    """

    start = tuple(board)
    if is_solved(start):
        return []

    queue = deque([start])
    parent: dict[tuple[int, ...], tuple[tuple[int, ...] | None, str | None]] = {
        start: (None, None)
    }
    last_move_for: dict[tuple[int, ...], str | None] = {start: None}

    while queue:
        current = queue.popleft()
        last_move = last_move_for[current]
        for move in legal_moves(current):
            if last_move is not None and move == _get_opposite(last_move):
                continue
            nxt = apply_move(current, move)
            if nxt in parent:
                continue
            parent[nxt] = (current, move)
            if nxt == GOAL:
                return _reconstruct(parent, nxt)
            last_move_for[nxt] = move
            queue.append(nxt)

    raise ValueError("board is not solvable")


def _reconstruct(
    parent: dict[tuple[int, ...], tuple[tuple[int, ...] | None, str | None]],
    end: tuple[int, ...],
) -> list[str]:
    path = []
    current = end
    while True:
        prev, move = parent[current]
        if prev is None:
            break
        assert move is not None
        path.append(move)
        current = prev
    path.reverse()
    return path
