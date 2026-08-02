"""Prompt and move-tag helpers."""

import re

from puzzle.render import render

VALID_MOVES = {"up", "down", "left", "right"}
MOVE_PATTERN = re.compile(r"<move>\s*(.*?)\s*</move>", re.IGNORECASE | re.DOTALL)


def parse_move(response_text: str) -> str | None:
    """Extract exactly one valid move tag."""

    matches = MOVE_PATTERN.findall(response_text)
    if len(matches) != 1:
        return None

    move = matches[0].strip().lower()
    return move if move in VALID_MOVES else None


def build_system_prompt() -> str:
    """Instruction shared by every model turn."""

    return (
        "You are a competitive puzzle solver. Make sure you read the puzzle "
        "instructions carefully, and always follow the required format.\n\n"
        "In each turn, think briefly inside <think>...</think> tags, then "
        "output exactly one move inside <move>...</move> tags."
    )


def build_board_prompt(board: tuple[int, ...]) -> str:
    """Prompt for the current board only."""

    return (
        "You are Player 0 in 15-puzzle.\n"
        "A 4x4 sliding puzzle board is given. The blank tile is shown as _.\n"
        "Your goal is to reach the solved board:\n"
        "1 2 3 4\n"
        "5 6 7 8\n"
        "9 10 11 12\n"
        "13 14 15 _\n\n"
        "At each turn, choose one legal move.\n"
        "Moves describe the numbered tile moving into the blank, not the blank moving.\n"
        "For example, if a row is 13 _ 14 15, then <move>left</move> means "
        "tile 14 moves left into the blank, producing 13 14 _ 15.\n\n"
        "Allowed moves are: up, down, left, right.\n"
        "Wrap your move in <move>...</move>, for example: <move>left</move>.\n\n"
        f"Current board:\n{render(board)}"
    )
