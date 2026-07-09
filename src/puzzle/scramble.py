import random

from puzzle.board import GOAL, apply_move, get_non_reversing_moves


def scramble_board(depth: int, rng: random.Random) -> tuple[int, ...]:
    """
    Scramble the board by starting from the GOAL state and taking 'depth' random legal moves.
    """

    board = GOAL

    last_move = None

    for _ in range(depth):
        moves = get_non_reversing_moves(board, last_move)

        move = rng.choice(moves)

        last_move = move

        board = apply_move(board, move)

    return board
