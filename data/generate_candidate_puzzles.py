import argparse
import json
import random
from typing import Any

from puzzle.scramble import scramble_board


def generate_candidates(
    start_depth: int, end_depth: int, num_per_depth: int, rng: random.Random
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

            generated_count += 1

            seen_boards.add(board)

            generated.append({"depth": depth, "board": board})

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

    args = parser.parse_args()

    if args.start_depth < 1:
        raise ValueError("start_depth must be at least 1")

    if args.end_depth < args.start_depth:
        raise ValueError("end_depth must be greater than or equal to start_depth")

    if args.num_per_depth <= 0:
        raise ValueError("num_per_depth must be positive")

    rng = random.Random(args.seed)

    candidates = generate_candidates(
        start_depth=args.start_depth,
        end_depth=args.end_depth,
        num_per_depth=args.num_per_depth,
        rng=rng,
    )

    write_candidates_jsonl(candidates, args.output)


if __name__ == "__main__":
    main()
