import argparse
import json
from typing import Any

from puzzle.render import render


def read_jsonl(input_path: str) -> list[dict[str, Any]]:
    """
    Read JSONL records from disk into a list of dictionaries.
    """

    records = []

    with open(input_path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            records.append(json.loads(line))

    return records


def build_system_message() -> dict[str, str]:
    """
    Build the fixed system instruction for puzzle SFT examples.
    """

    return {
        "role": "system",
        "content": (
            "You are a competitive puzzle solver. Make sure you read the puzzle "
            "instructions carefully, and always follow the required format.\n\n"
            "In each turn, think briefly inside <think>...</think> tags, then "
            "output exactly one move inside <move>...</move> tags."
        ),
    }


def build_initial_user_message(board: list[int]) -> dict[str, str]:
    """
    Build the initial user message with puzzle rules and the starting board.
    """

    return {
        "role": "user",
        "content": (
            "You are Player 0 in 15-puzzle.\n"
            "A 4x4 sliding puzzle board is given. The blank tile is shown as _.\n"
            "Your goal is to reach the solved board:\n"
            "1 2 3 4\n"
            "5 6 7 8\n"
            "9 10 11 12\n"
            "13 14 15 _\n\n"
            "At each turn, choose one legal move.\n"
            "Moves describe the numbered tile moving into the blank, not the blank moving.\n"
            "For example, if a row is 13 _ 14 15, then <move>left</move> means "
            "tile 14 moves left into the blank, producing 13 14 _ 15.\n\n"
            "Allowed moves are: up, down, left, right.\n"
            "Wrap your move in <move>...</move>, for example: <move>left</move>.\n\n"
            f"Current board:\n{render(tuple(board))}"
        ),
    }


def build_board_update_message(board: list[int]) -> dict[str, str]:
    """
    Build the next user message after an assistant move.
    """

    return {
        "role": "user",
        "content": f"Board after move:\n{render(tuple(board))}\n\nContinue.",
    }


def build_assistant_message(rationale: str, move: str) -> dict[str, str]:
    """
    Build one assistant SFT step using the rationale and known move.
    """

    return {
        "role": "assistant",
        "content": f"<think>\n{rationale}\n</think>\n<move>{move}</move>",
    }


def export_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Convert one rationale record into a final SFT training example.
    """

    prompt = [
        build_system_message(),
        build_initial_user_message(record["board"]),
    ]

    completion = []
    steps = record["steps"]

    for index, step in enumerate(steps):
        completion.append(
            build_assistant_message(
                rationale=step["rationale"],
                move=step["move"],
            )
        )

        if index < len(steps) - 1:
            completion.append(build_board_update_message(step["board_after"]))

    return {
        "prompt": prompt,
        "completion": completion,
        "answer": ",".join(record["optimal_moves"]),
        "reward": 1.0,
        "task": record["task"],
        "id": record["id"],
        "board": record["board"],
        "scramble_depth": record["scramble_depth"],
        "optimal_moves": record["optimal_moves"],
        "optimal_length": record["optimal_length"],
        "move_count": len(record["optimal_moves"]),
        "split": record.get("split", "train"),
    }


def write_jsonl_record(record: dict[str, Any], output_path: str) -> None:
    """
    Append one exported record to a JSONL file.
    """

    with open(output_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def export_dataset(records: list[dict[str, Any]], output_path: str) -> dict[str, int]:
    """
    Export all rationale records into final SFT JSONL rows.
    """

    exported = 0

    for record in records:
        write_jsonl_record(export_record(record), output_path)
        exported += 1

    return {"exported": exported}


def main() -> None:
    """
    Parse CLI arguments and export the rationale dataset into final SFT format.
    """

    parser = argparse.ArgumentParser(
        description="Export puzzle rationale records into final SFT JSONL format."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    records = read_jsonl(args.input)
    summary = export_dataset(records, args.output)
    print(f"Exported: {summary['exported']}")


if __name__ == "__main__":
    main()
