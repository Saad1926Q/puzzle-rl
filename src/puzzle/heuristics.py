from puzzle.board import GOAL, GOAL_POS, WIDTH, get_goal_rc, index_to_rc, rc_to_index


def _build_manhattan_table() -> tuple[tuple[int, ...], ...]:
    table = []

    for tile in range(len(GOAL)):
        goal_r, goal_c = GOAL_POS[tile]
        distances = []

        for idx in range(len(GOAL)):
            row, col = index_to_rc(idx)
            distances.append(abs(row - goal_r) + abs(col - goal_c))

        table.append(tuple(distances))

    return tuple(table)


MANHATTAN_TABLE = _build_manhattan_table()


def manhattan(board: tuple[int, ...]) -> int:
    """
    Sum of Manhattan distances of each tile from its goal position.
    """

    dist = 0

    for idx, tile in enumerate(board):
        if tile == 0:
            continue

        dist += MANHATTAN_TABLE[tile][idx]

    return dist


def row_linear_conflicts(board: tuple[int, ...]) -> int:
    conflicts = 0

    for row in range(WIDTH):
        goal_cols = []

        for col in range(WIDTH):
            tile = board[rc_to_index(row, col)]
            if tile == 0:
                continue

            goal_r, goal_c = get_goal_rc(tile)
            if goal_r == row:
                goal_cols.append(goal_c)

        for i in range(len(goal_cols)):
            for j in range(i + 1, len(goal_cols)):
                if goal_cols[i] > goal_cols[j]:
                    conflicts += 2

    return conflicts


def col_linear_conflicts(board: tuple[int, ...]) -> int:
    conflicts = 0

    for col in range(WIDTH):
        goal_rows = []

        for row in range(WIDTH):
            tile = board[rc_to_index(row, col)]
            if tile == 0:
                continue

            goal_r, goal_c = get_goal_rc(tile)
            if goal_c == col:
                goal_rows.append(goal_r)

        for i in range(len(goal_rows)):
            for j in range(i + 1, len(goal_rows)):
                if goal_rows[i] > goal_rows[j]:
                    conflicts += 2

    return conflicts


def manhattan_linear_conflict(board: tuple[int, ...]) -> int:
    return manhattan(board) + row_linear_conflicts(board) + col_linear_conflicts(board)
