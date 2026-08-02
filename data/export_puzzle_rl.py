import argparse
import json
from pathlib import Path
from typing import Any


def bucket_for(optimal_length: int) -> str:
    if 1 <= optimal_length <= 6:
        return "trivial"
    if 7 <= optimal_length <= 12:
        return "easy"
    if 13 <= optimal_length <= 24:
        return "medium"
    if 25 <= optimal_length <= 80:
        return "hard"
    raise ValueError(f"Optimal length {optimal_length} is outside 1-80")


def export_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    optimal_length = record["optimal_length"]

    return {
        "id": f"rl_{index:06d}",
        "board": record["board"],
        "bucket": bucket_for(optimal_length),
        "split": "rl",
        "scramble_depth": record["scramble_depth"],
        "optimal_moves": record["optimal_moves"],
        "optimal_length": optimal_length,
    }


def export_dataset(input_path: Path, output_path: Path) -> int:
    exported = 0

    with (
        input_path.open(encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as destination,
    ):
        for line in source:
            if not line.strip():
                continue

            record = json.loads(line)
            exported += 1
            destination.write(json.dumps(export_record(record, exported)) + "\n")

    return exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export generated 15-puzzle boards as an RL dataset."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    exported = export_dataset(args.input, args.output)
    print(f"Exported {exported} RL puzzles to {args.output}")


if __name__ == "__main__":
    main()
