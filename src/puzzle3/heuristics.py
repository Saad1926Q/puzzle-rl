from puzzle3.board import GOAL, GOAL_POS, index_to_rc


def manhattan(board: tuple[int, ...]) -> int:
    """Sum of Manhattan distances of each tile from its goal position."""

    dist = 0
    for idx, tile in enumerate(board):
        if tile == 0:
            continue
        row, col = index_to_rc(idx)
        goal_r, goal_c = GOAL_POS[tile]
        dist += abs(row - goal_r) + abs(col - goal_c)
    return dist


def misplaced_tiles(board: tuple[int, ...]) -> int:
    """Count non-blank tiles that are not in their goal position."""

    return sum(tile != 0 and tile != GOAL[idx] for idx, tile in enumerate(board))
