"""Experiment-owned constants for 8-puzzle SFT data generation."""

DEFAULT_TEACHER_MODEL = "meta/muse-spark-1.3-contributor"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_MAX_TURNS = 45
DEFAULT_THINKING = True

SFT_SOLVING_PROMPT = """You solve one 3x3 sliding puzzle one move at a time.
The solved board is 1 2 3 / 4 5 6 / 7 8 0. On each turn, slide one numbered
tile adjacent to the blank (0) into the blank by calling slide_tile exactly
once with that tile's number. Only a tile directly adjacent to the blank is a
legal move; the blank itself, a non-adjacent tile, or more than one tile per
turn is illegal and will be rejected.

Solve systematically in stages:
1. Put tile 1 in the top-left and preserve it.
2. Complete the top row by arranging tiles 2 and 3 as a pair. Reach the staging
   pattern with tile 2 in the top-right, tile 3 directly below it, and the blank
   in the top-middle; then slide tile 2 left and tile 3 up to form 1 2 3.
3. Preserve the completed top row. Complete the left column by arranging tiles
   4 and 7 as a pair. Reach the staging pattern with tile 4 in the bottom-left,
   tile 7 in the bottom-middle, and the blank in the middle-left; then slide
   tile 4 up and tile 7 left.
4. Preserve the completed top row and left column. Rotate the remaining
   lower-right 2x2 region until it is 5 6 / 8 0.

Do not greedily lock the last tile of a row or column without first positioning
its partner. Move the blank and unsolved tiles through the unsolved region to
reach each staging pattern. Track the current stage and a short multi-move
plan. On every turn, re-read the latest board, verify that the next planned
tile is adjacent to the blank, briefly state the immediate subgoal, and call
slide_tile exactly once. Avoid accidental immediate reversals and repeated
boards. If the observed board differs from the plan, discard the stale plan
and re-plan from the observed board. Keep reasoning concise."""

SFT_SOLVING_PROMPT_WITH_HISTORY = SFT_SOLVING_PROMPT + (
    "\nThe latest board observation is authoritative; retained history is "
    "supplementary context."
)
ANNOTATION_PROMPT = """You are annotating a verified 8-puzzle solution for SFT.
Explain why the already-selected move supports the current solving subgoal.
Use only the supplied boards, tile, and trajectory context. Do not choose a
different move, claim optimality, mention a solver, or mention reward or exact distance.
Return one concrete sentence of 10 to 30 words."""
