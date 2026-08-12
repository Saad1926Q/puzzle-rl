"""Constants for the 8-puzzle evaluation protocol and DeepSeek adapter."""

from __future__ import annotations

import re

VALID_MOVES = ("up", "down", "left", "right")
VALID_MOVE_SET = frozenset(VALID_MOVES)
MOVE_TAG_RE = re.compile(
    r"<move>\s*(up|down|left|right)\s*</move>",
    re.IGNORECASE,
)

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/beta"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_THINKING = True

SYSTEM_PROMPT = """You solve one 3x3 sliding puzzle one move at a time.
The numbered tile moves into the blank: `up` means the tile below the blank moves up;
`down` means the tile above the blank moves down; `left` means the tile right of the blank moves
left; and `right` means the tile left of the blank moves right.
Call submit_move exactly once with the single next move. Do not output a plan or a list of moves.
The board in the user message is the complete current state; do not assume or request history."""

MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_move",
        "description": "Submit exactly one next move for the current 8-puzzle board.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "enum": list(VALID_MOVES),
                    "description": "The one tile-moving direction to execute.",
                }
            },
            "required": ["move"],
            "additionalProperties": False,
        },
    },
}

# Rollout/scoring constants.
SOLVED_BASE_REWARD = 0.8
SOLVED_EFFICIENCY_WEIGHT = 0.2
ILLEGAL_OR_MALFORMED_REWARD = -0.1
TIMEOUT_REWARD = 0.0
MAX_TURNS = 120
DEFAULT_MAX_TURNS = MAX_TURNS
