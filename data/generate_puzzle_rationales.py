import argparse
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from puzzle.board import apply_move, legal_moves
from puzzle.render import render

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

load_dotenv()


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


def build_solution_steps(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Replay the known optimal move sequence and capture per-step board states.
    """

    current_board = tuple(candidate["board"])
    steps = []

    for step_index, move in enumerate(candidate["optimal_moves"]):
        if move not in legal_moves(current_board):
            raise ValueError(
                f"Illegal optimal move {move!r} at step {step_index} for candidate "
                f"{candidate.get('id', '<unknown>')}"
            )

        next_board = apply_move(current_board, move)
        steps.append(
            {
                "step_index": step_index,
                "board_before": list(current_board),
                "move": move,
                "board_after": list(next_board),
            }
        )
        current_board = next_board

    return steps


def build_rationale_prompt(step: dict[str, Any]) -> list[dict[str, str]]:
    """
    Build a prompt that asks for a concise rationale for one already-chosen move.
    """

    return [
        {
            "role": "system",
            "content": (
                "You are a strong 15-puzzle solver. "
                "Write concise, natural move rationales. "
                "Do not include XML tags, bullet points, or extra formatting. "
                "Do not mention alternative moves. "
                "Return only the short rationale text."
            ),
        },
        {
            "role": "user",
            "content": (
                "You are solving this 15-puzzle position.\n\n"
                f"Current board:\n{render(tuple(step['board_before']))}\n\n"
                f"Chosen move: {step['move']}\n"
                f"Board after the move:\n{render(tuple(step['board_after']))}\n\n"
                "Write the thought process behind making that move concisely, "
                "as a short player-style rationale."
            ),
        },
    ]


def call_deepseek(messages: list[dict[str, str]], model: str = DEFAULT_MODEL) -> str:
    """
    Call DeepSeek and return only the visible text response.
    """

    api_key = os.environ.get("DEEPSEEK_API_KEY")

    if api_key is None:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )

    content = response.choices[0].message.content

    if content is None:
        raise RuntimeError("DeepSeek returned an empty response")

    return content


def clean_rationale(text: str) -> str:
    """
    Normalize the returned rationale into a single compact string.
    """

    return " ".join(text.strip().split())


def make_rationale_record(
    candidate: dict[str, Any],
    steps: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    """
    Build the final success record for one puzzle with rationale annotations.
    """

    return {
        "status": "success",
        "task": "15_puzzle",
        "id": candidate["id"],
        "board": candidate["board"],
        "scramble_depth": candidate["scramble_depth"],
        "optimal_moves": candidate["optimal_moves"],
        "optimal_length": candidate["optimal_length"],
        "steps": steps,
        "source_model": model,
        "split": candidate.get("split", "train"),
    }


def generate_rationales_for_puzzle(
    candidate: dict[str, Any], model: str = DEFAULT_MODEL
) -> dict[str, Any]:
    """
    Generate concise per-step rationales for one solved puzzle trajectory.
    """

    try:
        steps = build_solution_steps(candidate)
    except Exception as exc:
        return {
            "status": "failure",
            "task": "15_puzzle",
            "candidate": candidate,
            "error": str(exc),
        }

    for step in steps:
        try:
            response_text = call_deepseek(
                build_rationale_prompt(step),
                model=model,
            )
        except Exception as exc:
            return {
                "status": "failure",
                "task": "15_puzzle",
                "candidate": candidate,
                "step_index": step["step_index"],
                "error": str(exc),
            }

        rationale = clean_rationale(response_text)

        if not rationale:
            return {
                "status": "failure",
                "task": "15_puzzle",
                "candidate": candidate,
                "step_index": step["step_index"],
                "error": "DeepSeek returned an empty rationale",
            }

        step["rationale"] = rationale

    return make_rationale_record(candidate, steps, model)


def write_jsonl_record(record: dict[str, Any], output_path: str) -> None:
    """
    Append one generated record to a JSONL output file.
    """

    with open(output_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def generate_rationale_dataset(
    candidates: list[dict[str, Any]],
    output_path: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, int]:
    """
    Generate rationale records for all solved puzzles and write each attempt to disk.
    """

    attempted = 0
    kept = 0
    failed = 0

    progress = tqdm(candidates, desc="Generating rationales", unit="puzzle")

    for candidate in progress:
        attempted += 1
        result = generate_rationales_for_puzzle(candidate, model=model)
        write_jsonl_record(result, output_path)

        if result["status"] == "failure":
            failed += 1
            continue

        kept += 1

    return {"attempted": attempted, "kept": kept, "failed": failed}


def main() -> None:
    """
    Parse CLI arguments and run rationale generation for solved puzzle records.
    """

    parser = argparse.ArgumentParser(
        description="Generate concise rationale annotations for solved 15-puzzle trajectories."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise ValueError(f"Input file does not exist: {args.input}")

    candidates = read_jsonl(args.input)

    summary = generate_rationale_dataset(
        candidates=candidates,
        output_path=args.output,
        model=args.model,
    )

    print(f"Succeeded: {summary['kept']}, failed: {summary['failed']}")


if __name__ == "__main__":
    main()
