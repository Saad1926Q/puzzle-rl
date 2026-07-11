import argparse
import json
import os
import re
from typing import Any

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from puzzle.board import apply_move, is_solved, legal_moves
from puzzle.render import render

VALID_MOVES = {"up", "down", "left", "right"}
MOVE_PATTERN = re.compile(r"<move>\s*(.*?)\s*</move>", re.IGNORECASE | re.DOTALL)


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


def load_hf_model(model_name_or_path: str):
    """
    Load a Qwen3 chat model and tokenizer for local evaluation.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
    )
    model.eval()

    return tokenizer, model


def call_model(
    messages: list[dict[str, str]],
    tokenizer,
    model,
) -> str:
    """
    Run one chat completion step with a local Qwen3 model.
    """

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return response_text


def compute_reward(
    solved: bool,
    steps_taken: int,
    illegal_move: bool,
    format_failure: bool,
    optimal_length: int | None,
) -> float:
    """
    Compute the evaluation reward for one completed trajectory.
    """

    reward = 0.0

    if solved:
        reward += 1.0

    if illegal_move or format_failure:
        reward -= 0.25

    if optimal_length is not None:
        reward -= 0.01 * max(0, steps_taken - optimal_length)

    return reward


def run_one_trajectory(
    candidate: dict[str, Any],
    tokenizer,
    model,
    max_steps: int,
) -> dict[str, Any]:
    """
    Run one full multi-turn rollout for a single eval puzzle.
    """

    current_board = tuple(candidate["board"])
    prompt = build_initial_prompt(current_board)
    completion: list[dict[str, str]] = []
    moves: list[str] = []
    response_text: str | None = None
    solved = False
    illegal_move = False
    format_failure = False

    for _ in range(max_steps):
        response_text = call_model(prompt + completion, tokenizer, model)
        move = parse_move(response_text)

        if move is None:
            format_failure = True
            break

        next_board = verify_and_apply_move(current_board, move)

        if next_board is None:
            illegal_move = True
            break

        completion.append({"role": "assistant", "content": response_text})
        moves.append(move)
        current_board = next_board

        if is_solved(current_board):
            solved = True
            break

        completion.append(build_board_update_message(current_board))

    reward = compute_reward(
        solved=solved,
        steps_taken=len(moves),
        illegal_move=illegal_move,
        format_failure=format_failure,
        optimal_length=candidate.get("optimal_length"),
    )

    return {
        "status": "success" if solved else "failure",
        "solved": solved,
        "bucket": candidate["bucket"],
        "board": candidate["board"],
        "scramble_depth": candidate["scramble_depth"],
        "optimal_moves": candidate.get("optimal_moves", []),
        "optimal_length": candidate.get("optimal_length"),
        "moves": moves,
        "move_count": len(moves),
        "illegal_move": illegal_move,
        "format_failure": format_failure,
        "reward": reward,
        "completion": completion,
        "last_response": response_text,
    }


def run_eval_for_puzzle(
    candidate: dict[str, Any],
    tokenizer,
    model,
    max_steps: int,
    num_generations: int,
) -> list[dict[str, Any]]:
    """
    Run multiple evaluation rollouts for one puzzle.
    """

    return [
        run_one_trajectory(candidate, tokenizer, model, max_steps)
        for _ in range(num_generations)
    ]


def write_jsonl_record(record: dict[str, Any], output_path: str) -> None:
    """
    Append one eval result record to a JSONL output file.
    """

    with open(output_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def main() -> None:
    """
    Parse CLI arguments, run eval rollouts, and write raw results.
    """

    parser = argparse.ArgumentParser(
        description="Run local Qwen3 eval rollouts for 15-puzzle."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-generations", type=int, default=1)

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise ValueError(f"Input file does not exist: {args.input}")

    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")

    if args.num_generations <= 0:
        raise ValueError("num_generations must be positive")

    candidates = read_jsonl(args.input)
    tokenizer, model = load_hf_model(args.model)

    with open(args.output, "w", encoding="utf-8"):
        pass

    progress = tqdm(candidates, desc="Running eval", unit="puzzle")

    for candidate in progress:
        puzzle_results = run_eval_for_puzzle(
            candidate=candidate,
            tokenizer=tokenizer,
            model=model,
            max_steps=args.max_steps,
            num_generations=args.num_generations,
        )

        for result in puzzle_results:
            write_jsonl_record(result, args.output)


if __name__ == "__main__":
    main()
