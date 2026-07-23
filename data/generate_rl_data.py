# We want 10,000 unique RL puzzles with known optimal lengths from 1 to 80.
# Any boards already used for SFT or eval are excluded.
#
# For lengths 1-12, we can run BFS from the solved board and enumerate every
# possible board. Since BFS visits shorter paths first, the BFS layer gives us
# the exact optimal length. From these layers, we select 3,000 boards:
#   - all 146 remaining boards at lengths 1-6
#   - 2,854 boards at lengths 7-12
#
# BFS gets too large after this point. We generate random candidates, solve them
# optimally, and keep 4,000 medium boards at lengths 13-24 and 3,000 hard boards
# at lengths 25-80.
#
# SHALLOW_TARGETS_BY_DISTANCE contains the exact number wanted at lengths 1-12.

import json
import random
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from puzzle.board import GOAL, _get_opposite, apply_move, legal_moves
from puzzle.scramble import scramble_board
from puzzle.solver import solve

SFT_PATH = Path("data/sft_puzzles_1000.jsonl")
EVAL_PATH = Path("data/eval_puzzles_100.jsonl")
OUTPUT_PATH = Path("data/rl_puzzles.jsonl")
SHALLOW_OUTPUT_PATH = Path("data/rl_puzzles_1_12.jsonl")

SEED = 43
WORKERS = 6

TARGET_COUNTS = {
    "trivial": 146,
    "easy": 2_854,
    "medium": 4_000,
    "hard": 3_000,
}

SCRAMBLE_DEPTH_RANGES = {
    "medium": (15, 25),
    "hard": (30, 80),
}

SHALLOW_TARGETS_BY_DISTANCE = {
    1: 2,
    2: 4,
    3: 10,
    4: 20,
    5: 54,
    6: 56,
    7: 159,
    8: 390,
    9: 577,
    10: 576,
    11: 576,
    12: 576,
}

Board = tuple[int, ...]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def load_boards(path: Path) -> set[tuple[int, ...]]:
    return {tuple(record["board"]) for record in read_jsonl(path)}


def generate_trivial_easy(
    excluded_boards: set[Board],
    rng: random.Random,
    targets: Mapping[int, int] = SHALLOW_TARGETS_BY_DISTANCE,
) -> list[dict[str, Any]]:
    """Generate the configured trivial and easy boards with exact BFS solutions."""

    seen: set[Board] = {GOAL}
    parents: dict[Board, tuple[Board | None, str | None]] = {GOAL: (None, None)}
    current_layer: list[Board] = [GOAL]
    candidates = []

    with tqdm(
        total=sum(targets.values()), desc="Trivial/easy", unit="puzzle"
    ) as progress:
        for depth in range(1, max(targets, default=0) + 1):
            next_layer: list[Board] = []

            for board in current_layer:
                for move in legal_moves(board):
                    child = apply_move(board, move)

                    if child in seen:
                        continue

                    seen.add(child)
                    parents[child] = (board, move)
                    next_layer.append(child)

            current_layer = next_layer
            available = [
                board for board in current_layer if board not in excluded_boards
            ]
            target = targets.get(depth, 0)

            if target > len(available):
                raise ValueError(
                    f"Optimal length {depth} needs {target} boards, "
                    f"but only {len(available)} remain after exclusions"
                )

            for board in rng.sample(available, target):
                optimal_moves = []
                current = board

                while current != GOAL:
                    parent, move_from_parent = parents[current]

                    if parent is None or move_from_parent is None:
                        raise ValueError("Incomplete BFS parent chain")

                    optimal_moves.append(_get_opposite(move_from_parent))
                    current = parent

                candidates.append(
                    {
                        "board": board,
                        "scramble_depth": depth,
                        "optimal_moves": optimal_moves,
                        "optimal_length": depth,
                    }
                )

            progress.update(target)

    return candidates


def _solve_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point that adds an optimal solution to one candidate."""

    optimal_moves = solve(tuple(candidate["board"]))

    return {
        **candidate,
        "optimal_moves": optimal_moves,
        "optimal_length": len(optimal_moves),
    }


def generate_solved_candidate(
    scramble_depth: int, rng: random.Random
) -> dict[str, Any]:
    """Generate one scrambled board and certify its optimal solution."""

    if scramble_depth < 1:
        raise ValueError("scramble_depth must be positive")

    candidate = {
        "board": scramble_board(scramble_depth, rng),
        "scramble_depth": scramble_depth,
    }
    return _solve_candidate(candidate)


def generate_medium_hard(
    excluded_boards: set[Board],
    rng: random.Random,
    medium_target: int = TARGET_COUNTS["medium"],
    hard_target: int = TARGET_COUNTS["hard"],
    workers: int = WORKERS,
) -> list[dict[str, Any]]:
    """Generate unique medium and hard candidates until both quotas are full."""

    if medium_target < 0 or hard_target < 0:
        raise ValueError("medium and hard targets must be non-negative")
    if workers < 1:
        raise ValueError("workers must be positive")

    remaining_medium = medium_target
    remaining_hard = hard_target
    seen_boards = set(excluded_boards)
    accepted = []
    accepted_medium = 0
    accepted_hard = 0
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None

    try:
        with tqdm(
            total=medium_target + hard_target, desc="Medium/hard", unit="puzzle"
        ) as progress:
            while remaining_medium > 0 or remaining_hard > 0:
                wave_size = min(
                    workers * 4,
                    remaining_medium + remaining_hard,
                )
                raw_candidates = []

                while len(raw_candidates) < wave_size:
                    if remaining_medium > 0:
                        minimum, maximum = SCRAMBLE_DEPTH_RANGES["medium"]
                    else:
                        minimum, maximum = SCRAMBLE_DEPTH_RANGES["hard"]

                    scramble_depth = rng.randint(minimum, maximum)
                    board = scramble_board(scramble_depth, rng)

                    if board in seen_boards:
                        continue

                    seen_boards.add(board)
                    raw_candidates.append(
                        {
                            "board": board,
                            "scramble_depth": scramble_depth,
                        }
                    )

                if executor is None:
                    solved_candidates = map(_solve_candidate, raw_candidates)
                else:
                    solved_candidates = executor.map(
                        _solve_candidate,
                        raw_candidates,
                        chunksize=1,
                    )

                for candidate in solved_candidates:
                    optimal_length = candidate["optimal_length"]

                    if 13 <= optimal_length <= 24 and remaining_medium > 0:
                        remaining_medium -= 1
                        accepted_medium += 1
                    elif 25 <= optimal_length <= 80 and remaining_hard > 0:
                        remaining_hard -= 1
                        accepted_hard += 1
                    else:
                        continue

                    accepted.append(candidate)
                    progress.update(1)
                    progress.set_postfix(
                        medium=f"{accepted_medium}/{medium_target}",
                        hard=f"{accepted_hard}/{hard_target}",
                        refresh=False,
                    )
    finally:
        if executor is not None:
            executor.shutdown()

    return accepted


def main() -> None:
    excluded_boards = load_boards(SFT_PATH) | load_boards(EVAL_PATH)
    rng = random.Random(SEED)

    trivial_easy = generate_trivial_easy(
        excluded_boards=excluded_boards,
        rng=rng,
    )
    write_jsonl(trivial_easy, SHALLOW_OUTPUT_PATH)
    print(f"Wrote {len(trivial_easy)} shallow puzzles to {SHALLOW_OUTPUT_PATH}")

    medium_hard_exclusions = excluded_boards | {
        tuple(candidate["board"]) for candidate in trivial_easy
    }
    medium_hard = generate_medium_hard(
        excluded_boards=medium_hard_exclusions,
        rng=rng,
    )

    candidates = trivial_easy + medium_hard
    write_jsonl(candidates, OUTPUT_PATH)
    print(f"Wrote {len(candidates)} RL puzzles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
