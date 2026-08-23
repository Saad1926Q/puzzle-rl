from collections import deque
from threading import Lock

from puzzle3.board import (
    GOAL,
    Board,
    TileAction,
    adjacent_tiles,
    is_solved,
    slide_tile,
)


_DISTANCE_TABLE: dict[Board, int] | None = None
_DISTANCE_TABLE_LOCK = Lock()


def _build_distance_table() -> dict[Board, int]:
    """Build exact distances from every reachable board to ``GOAL``."""

    distances = {GOAL: 0}
    queue = deque([GOAL])
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for tile in adjacent_tiles(current):
            nxt = slide_tile(current, tile)
            if nxt not in distances:
                distances[nxt] = next_distance
                queue.append(nxt)
    return distances


def exact_distance(board: Board) -> int:
    """Return the exact optimal number of actions from ``board`` to ``GOAL``.

    The complete reachable-state table is built lazily and shared by all
    rollouts. This avoids running a fresh BFS for every evaluated transition.
    """

    global _DISTANCE_TABLE
    if _DISTANCE_TABLE is None:
        with _DISTANCE_TABLE_LOCK:
            if _DISTANCE_TABLE is None:
                _DISTANCE_TABLE = _build_distance_table()
    try:
        return _DISTANCE_TABLE[tuple(board)]
    except KeyError as exc:
        raise ValueError("board is not a solvable 8-puzzle state") from exc


def solve(board: Board) -> list[TileAction]:
    """
    Solve an 8-puzzle board optimally using BFS.

    The 8-puzzle has only 181,440 solvable states, so BFS is simple and robust
    for eval-set generation and diagnostics.
    """

    start = tuple(board)
    if is_solved(start):
        return []

    queue = deque([start])
    parent: dict[Board, tuple[Board | None, TileAction | None]] = {start: (None, None)}

    while queue:
        current = queue.popleft()
        for tile in adjacent_tiles(current):
            nxt = slide_tile(current, tile)
            if nxt in parent:
                continue
            parent[nxt] = (current, tile)
            if nxt == GOAL:
                return _reconstruct(parent, nxt)
            queue.append(nxt)

    raise ValueError("board is not solvable")


def _reconstruct(
    parent: dict[Board, tuple[Board | None, TileAction | None]],
    end: Board,
) -> list[TileAction]:
    path: list[TileAction] = []
    current = end
    while True:
        prev, tile = parent[current]
        if prev is None:
            break
        assert tile is not None
        path.append(tile)
        current = prev
    path.reverse()
    return path
