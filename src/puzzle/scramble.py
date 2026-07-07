import random

from puzzle.board import GOAL, apply_move, legal_moves


def scramble_board(depth: int, rng: random.Random) -> tuple[int, ...]:
    """
    Scramble the board by starting from the GOAL state and taking 'depth' random legal moves.
    """

    board = GOAL

    for _ in range(depth):
        moves = legal_moves(board)

        move = rng.choice(moves)

        board = apply_move(board, move)

    return board
