import random

from puzzle3.board import Board, GOAL, TileAction, adjacent_tiles, slide_tile


def scramble_board(depth: int, rng: random.Random) -> Board:
    """Scramble from GOAL without immediately reversing an action."""

    board = GOAL
    previous_tile: TileAction | None = None
    for _ in range(depth):
        choices = [tile for tile in adjacent_tiles(board) if tile != previous_tile]
        tile = rng.choice(choices)
        previous_tile = tile
        board = slide_tile(board, tile)
    return board
