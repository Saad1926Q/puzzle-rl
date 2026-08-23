"""Constants for the 8-puzzle evaluation protocol and DeepSeek adapter."""

from __future__ import annotations

VALID_MOVES = ("up", "down", "left", "right")
VALID_MOVE_SET = frozenset(VALID_MOVES)

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/beta"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_GLM_MODEL = "glm-4.5-air"
DEFAULT_GLM_BASE_URL = "https://api.z.ai/api/paas/v4"
DEFAULT_GLM_API_KEY_ENV = "ZAI_API_KEY"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_THINKING = True

SYSTEM_PROMPT = """You solve one 3x3 sliding puzzle one move at a time.
On each turn, choose one numbered tile adjacent to the blank (0) and slide that tile into the
blank. Call slide_tile exactly once with that tile's number. Do not submit the blank, a
non-adjacent tile, a plan, or a list of tiles. The board in the user message is the complete
current state; do not assume or request history."""

MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "slide_tile",
        "description": "Slide one numbered tile adjacent to the blank into the blank.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "tile": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "description": "The adjacent numbered tile to slide into the blank.",
                }
            },
            "required": ["tile"],
            "additionalProperties": False,
        },
    },
}

# Rollout/scoring constants.
SOLVED_BASE_REWARD = 0.8
SOLVED_EFFICIENCY_WEIGHT = 0.2
ILLEGAL_OR_MALFORMED_REWARD = -1.0
TIMEOUT_REWARD = -0.25

# Reward legal transitions according to their change in exact solution distance.
DISTANCE_PROGRESS_WEIGHT = 0.25
MAX_PUZZLE_DISTANCE = 31
REWARD_SCHEME = "exact_distance_progress_v1"
ACTION_INTERFACE = "tile_id_v1"

MAX_TURNS = 45
DEFAULT_MAX_TURNS = MAX_TURNS
