"""3x3 sliding-puzzle primitives and DeepSeek evaluation harness."""

from puzzle3.board import GOAL, apply_move, is_solved, legal_moves
from puzzle3.dataset import PuzzleExample, load_examples
from puzzle3.evaluator import evaluate, evaluate_episode, solved_reward

__all__ = [
    "GOAL",
    "PuzzleExample",
    "apply_move",
    "evaluate",
    "evaluate_episode",
    "is_solved",
    "legal_moves",
    "load_examples",
    "solved_reward",
]
