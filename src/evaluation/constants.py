"""Constants for the 8-puzzle evaluation protocol and provider adapters."""

from __future__ import annotations


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/beta"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_GLM_MODEL = "glm-4.5-air"
DEFAULT_GLM_BASE_URL = "https://api.z.ai/api/paas/v4"
DEFAULT_GLM_API_KEY_ENV = "ZAI_API_KEY"
DEFAULT_CROF_MODEL = "glm-5.3-flash"
DEFAULT_CROF_BASE_URL = "https://crof.ai/v1"
DEFAULT_CROF_API_KEY_ENV = "CROF_API_KEY"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_QWEN_BASE_URL = "http://localhost:8000/v1"
DEFAULT_QWEN_TEMPERATURE = 1.0
DEFAULT_QWEN_TOP_P = 1.0
DEFAULT_QWEN_TOP_K = 20
DEFAULT_QWEN_PRESENCE_PENALTY = 2.0
DEFAULT_QWEN_REPETITION_PENALTY = 1.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_THINKING = True

FIXED_SYSTEM_PROMPT = """You solve one 3x3 sliding puzzle one move at a time.
The solved board is 1 2 3 / 4 5 6 / 7 8 0. On each turn, slide one numbered
tile adjacent to the blank (0) into the blank by calling slide_tile exactly
once with that tile's number. Only a tile directly adjacent to the blank is a
legal move; the blank itself, a non-adjacent tile, or more than one tile per
turn is illegal and will be rejected."""

SEED_STRATEGY_PROMPT = """Think concisely about the current move, then call
slide_tile. Treat the latest board returned by the environment as authoritative;
retained history is supplementary context."""


def build_system_prompt(strategy_prompt: str = SEED_STRATEGY_PROMPT) -> str:
    """Combine immutable environment rules with mutable strategy guidance."""

    strategy = strategy_prompt.strip()
    if not strategy:
        raise ValueError("strategy_prompt must not be empty")
    return f"{FIXED_SYSTEM_PROMPT}\n{strategy}"


SYSTEM_PROMPT = build_system_prompt()
SYSTEM_PROMPT_WITH_HISTORY = SYSTEM_PROMPT

SLIDE_TILE_TOOL = {
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
