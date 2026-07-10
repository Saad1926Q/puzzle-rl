import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from puzzle.scramble import scramble_board
from puzzle.solver import solve
from tqdm import tqdm


def generate_candidates(
    start_depth: int,
    end_depth: int,
    num_per_depth: int,
    rng: random.Random,
    excluded_boards: set[tuple[int, ...]],
) -> list[dict[str, Any]]:
    """
    Generate unique candidate puzzles across the requested scramble-depth range.
    """

    def _generate_candidates_for_depth(
        depth: int, num_candidates: int, rng: random.Random, seen_boards: set
    ) -> list[dict[str, Any]]:
        """
        Generate unique candidate puzzles for one exact scramble depth.
        """

        generated_count = 0

        generated = []

        while generated_count < num_candidates:
            board = scramble_board(depth=depth, rng=rng)

            if board in seen_boards:
                continue

            if board in excluded_boards:
                continue

            generated_count += 1

            seen_boards.add(board)

            generated.append({"scramble_depth": depth, "board": list(board)})

            if generated_count == num_candidates:
                break

        return generated

    seen_boards = set()

    all_candidates = []

    for i in range(start_depth, end_depth + 1):
        generated_candidates = _generate_candidates_for_depth(
            depth=i, num_candidates=num_per_depth, rng=rng, seen_boards=seen_boards
        )

        all_candidates.extend(generated_candidates)

    return all_candidates


def _exclude_eval_puzzles(eval_path: str | None) -> set[tuple[int, ...]]:
    """
    Load exact eval boards that should be excluded from SFT generation.
    """

    if eval_path is None:
        return set()

    excluded_boards = set()

    with open(eval_path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            excluded_boards.add(tuple(record["board"]))

    return excluded_boards


def _solve_candidate(record: dict[str, Any]) -> dict[str, Any]:
    """
    Add optimal solution to a generated board record.
    """

    optimal_moves = solve(record["board"])

    return {
        "board": record["board"],
        "scramble_depth": record["scramble_depth"],
        "optimal_moves": optimal_moves,
        "optimal_length": len(optimal_moves),
    }


def write_candidates_jsonl(records: list[dict[str, Any]], output_path: str) -> None:
    """
    Write candidate puzzle records to a JSONL file.
    """

    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def main():
    """
    Parse CLI arguments and write the generated candidate puzzle set.
    """

    parser = argparse.ArgumentParser(
        description="Generate unique candidate 15-puzzle boards by scramble depth."
    )

    parser.add_argument("--start-depth", type=int, required=True)
    parser.add_argument("--end-depth", type=int, required=True)
    parser.add_argument("--num-per-depth", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()

    if args.start_depth < 1:
        raise ValueError("start_depth must be at least 1")

    if args.end_depth < args.start_depth:
        raise ValueError("end_depth must be greater than or equal to start_depth")

    if args.num_per_depth <= 0:
        raise ValueError("num_per_depth must be positive")

    if args.workers <= 0:
        raise ValueError("workers must be positive")

    rng = random.Random(args.seed)
    excluded_boards = _exclude_eval_puzzles(args.exclude)

    candidates = generate_candidates(
        start_depth=args.start_depth,
        end_depth=args.end_depth,
        num_per_depth=args.num_per_depth,
        rng=rng,
        excluded_boards=excluded_boards,
    )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        solved_candidates = list(
            tqdm(
                executor.map(_solve_candidate, candidates),
                total=len(candidates),
                desc="Solving puzzles",
                unit="puzzle",
            )
        )

    records = []
    for index, candidate in enumerate(solved_candidates, start=1):
        records.append(
            {
                "id": f"sft_{index:06d}",
                "board": candidate["board"],
                "scramble_depth": candidate["scramble_depth"],
                "optimal_moves": candidate["optimal_moves"],
                "optimal_length": candidate["optimal_length"],
                "split": "train",
            }
        )

    write_candidates_jsonl(records, args.output)


if __name__ == "__main__":
    main()
