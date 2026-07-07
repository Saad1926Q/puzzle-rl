from puzzle.board import GOAL, get_goal_rc, index_to_rc


def manhattan(board: tuple[int, ...]) -> int:
    """
    Sum of Manhattan distances of each tile from its goal position.
    """

    dist = 0

    for idx, element in enumerate(board):
        if element == 0:
            continue

        r, c = index_to_rc(idx)
        goal_r, goal_c = get_goal_rc(element)

        dist += abs(r - goal_r) + abs(c - goal_c)

    return dist
