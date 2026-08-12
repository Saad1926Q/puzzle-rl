import random

from puzzle3.board import GOAL, apply_move, get_non_reversing_moves


def scramble_board(depth: int, rng: random.Random) -> tuple[int, ...]:
    """Scramble from GOAL by taking depth random non-immediate-reversing moves."""

    board = GOAL
    last_move = None
    for _ in range(depth):
        move = rng.choice(get_non_reversing_moves(board, last_move))
        last_move = move
        board = apply_move(board, move)
    return board
