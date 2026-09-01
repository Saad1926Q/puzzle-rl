"""Prompt-optimization-only puzzle protocol and rollout constants."""

from __future__ import annotations

FIXED_PROTOCOL_PROMPT = """You solve one 3x3 sliding puzzle one move at a time.
The solved board is 1 2 3 / 4 5 6 / 7 8 0. On each turn, slide one numbered
tile adjacent to the blank (0) into the blank by calling slide_tile exactly
once with that tile's number. Only a tile directly adjacent to the blank is a
legal move; the blank itself, a non-adjacent tile, or more than one tile per
turn is illegal and will be rejected."""

SEED_STRATEGY_PROMPT = """Think concisely about the current move, then call
slide_tile. Treat the latest board returned by the environment as authoritative;
retained history is supplementary context."""

SLIDE_TILE_TOOL = {
    "type": "function",
    "function": {
        "name": "slide_tile",
        "description": "Slide one numbered tile adjacent to the blank into the blank.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"tile": {"type": "integer", "minimum": 1, "maximum": 8}},
            "required": ["tile"],
            "additionalProperties": False,
        },
    },
}

SOLVED_BASE_REWARD = 0.8
SOLVED_EFFICIENCY_WEIGHT = 0.2
ILLEGAL_OR_MALFORMED_REWARD = -1.0
TIMEOUT_REWARD = -0.25
DISTANCE_PROGRESS_WEIGHT = 0.25
MAX_PUZZLE_DISTANCE = 31
MAX_TURNS = 45
DEFAULT_MAX_TOKENS = 4096
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_THINKING = True
DEFAULT_OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


def build_candidate_system_prompt(strategy_prompt: str) -> str:
    """Combine the experiment's immutable protocol and mutable strategy text."""

    strategy = strategy_prompt.strip()
    if not strategy:
        raise ValueError("strategy_prompt must not be empty")
    return f"{FIXED_PROTOCOL_PROMPT}\n{strategy}"
