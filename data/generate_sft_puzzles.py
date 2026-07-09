import argparse
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from puzzle.board import apply_move, is_solved, legal_moves
from puzzle.render import render

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
VALID_MOVES = {"up", "down", "left", "right"}
MOVE_PATTERN = re.compile(r"<move>\s*(.*?)\s*</move>", re.IGNORECASE | re.DOTALL)

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


def build_initial_prompt(board: tuple[int, ...]) -> list[dict[str, str]]:
    """
    Build the initial chat prompt for one puzzle trajectory.
    """

    return [
        {
            "role": "system",
            "content": (
                "You are a competitive puzzle solver. Read the puzzle carefully "
                "and always follow the required format.\n\n"
                "At each turn, think normally and then output exactly one move "
                "inside <move>...</move> tags in the final answer."
            ),
        },
        {
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
                f"Current board:\n{render(board)}"
            ),
        },
    ]


def build_board_update_message(board: tuple[int, ...]) -> dict[str, str]:
    """
    Build the next user message after applying a verified move.
    """

    return {
        "role": "user",
        "content": f"Board after move:\n{render(board)}\n\nContinue.",
    }


def parse_move(response_text: str) -> str | None:
    """
    Extract one move tag from the model response and validate its value.
    """

    matches = MOVE_PATTERN.findall(response_text)

    if len(matches) != 1:
        return None

    move = matches[0].strip().lower()

    if move not in VALID_MOVES:
        return None

    return move


def verify_and_apply_move(board: tuple[int, ...], move: str) -> tuple[int, ...] | None:
    """
    Check that a move is legal for the board, then apply it.
    """

    if move not in legal_moves(board):
        return None

    return apply_move(board, move)


def call_deepseek(
    messages: list[dict[str, str]], model: str = DEFAULT_MODEL
) -> dict[str, str]:
    """
    Call DeepSeek and return the visible response plus reasoning content.
    """

    api_key = os.environ.get("DEEPSEEK_API_KEY")

    if api_key is None:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )

    content = response.choices[0].message.content

    if content is None:
        raise RuntimeError("DeepSeek returned an empty response")

    reasoning_content = response.choices[0].message.reasoning_content

    return {
        "content": content,
        "reasoning_content": reasoning_content or "",
    }


def make_sft_record(
    candidate: dict[str, Any],
    prompt: list[dict[str, str]],
    completion: list[dict[str, str]],
    moves: list[str],
    reasoning_steps: list[str],
    model: str,
) -> dict[str, Any]:
    """
    Build the final success record for one solved SFT trajectory.
    """

    board = tuple(candidate["board"])

    return {
        "status": "success",
        "prompt": prompt,
        "completion": completion,
        "answer": ",".join(moves),
        "reward": 1.0,
        "task": "15_puzzle",
        "board": list(board),
        "scramble_depth": candidate["depth"],
        "moves": moves,
        "move_count": len(moves),
        "reasoning_steps": reasoning_steps,
        "source_model": model,
    }


def generate_trajectory(
    candidate: dict[str, Any], model: str = DEFAULT_MODEL, max_steps: int = 100
) -> dict[str, Any]:
    """
    Generate one verified step-by-step trajectory for a single candidate puzzle.
    """

    current_board = tuple(candidate["board"])
    prompt = build_initial_prompt(current_board)
    completion = []
    moves = []
    reasoning_steps = []
    response_text = None
    reasoning_text = ""

    for _ in range(max_steps):
        response = call_deepseek(prompt + completion, model)
        response_text = response["content"]
        reasoning_text = response["reasoning_content"]
        move = parse_move(response_text)

        if move is None:
            return {
                "status": "failure",
                "candidate": candidate,
                "prompt": prompt,
                "completion": completion,
                "moves": moves,
                "current_board": list(current_board),
                "last_response": response_text,
                "reasoning_steps": reasoning_steps,
                "last_reasoning": reasoning_text,
            }

        next_board = verify_and_apply_move(current_board, move)

        if next_board is None:
            return {
                "status": "failure",
                "candidate": candidate,
                "prompt": prompt,
                "completion": completion,
                "moves": moves,
                "current_board": list(current_board),
                "last_response": response_text,
                "reasoning_steps": reasoning_steps,
                "last_reasoning": reasoning_text,
            }

        completion.append({"role": "assistant", "content": response_text})
        moves.append(move)
        reasoning_steps.append(reasoning_text)
        current_board = next_board

        if is_solved(current_board):
            return make_sft_record(
                candidate, prompt, completion, moves, reasoning_steps, model
            )

        completion.append(build_board_update_message(current_board))

    return {
        "status": "failure",
        "candidate": candidate,
        "prompt": prompt,
        "completion": completion,
        "moves": moves,
        "current_board": list(current_board),
        "last_response": response_text,
        "reasoning_steps": reasoning_steps,
        "last_reasoning": reasoning_text,
    }


def write_jsonl_record(record: dict[str, Any], output_path: str) -> None:
    """
    Append one generated record to a JSONL output file.
    """

    with open(output_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def generate_sft_dataset(
    candidates: list[dict[str, Any]],
    output_path: str,
    model: str = DEFAULT_MODEL,
    max_steps: int = 100,
) -> dict[str, int]:
    """
    Generate SFT records for all candidate puzzles and write each attempt to disk.
    """

    attempted = 0
    kept = 0
    failed = 0

    progress = tqdm(candidates, desc="Generating SFT", unit="puzzle")

    for candidate in progress:
        attempted += 1
        result = generate_trajectory(candidate, model=model, max_steps=max_steps)
        write_jsonl_record(result, output_path)

        if result["status"] == "failure":
            failed += 1
            continue

        kept += 1

    return {"attempted": attempted, "kept": kept, "failed": failed}


def main() -> None:
    """
    Parse CLI arguments and run SFT trajectory generation for the input dataset.
    """

    parser = argparse.ArgumentParser(
        description="Generate SFT 15-puzzle trajectories with DeepSeek."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-steps", type=int, default=100)

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise ValueError(f"Input file does not exist: {args.input}")

    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")

    candidates = read_jsonl(args.input)

    summary = generate_sft_dataset(
        candidates=candidates,
        output_path=args.output,
        model=args.model,
        max_steps=args.max_steps,
    )

    print(
        f"Succeeded: {summary['kept']}, failed: {summary['failed']}"
    )


if __name__ == "__main__":
    main()
