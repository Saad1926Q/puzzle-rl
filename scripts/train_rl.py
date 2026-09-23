"""Train the 8-puzzle policy with bounded-history asynchronous GRPO."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer

from puzzle3.environment import (
    DEFAULT_HISTORY_TURNS,
    DEFAULT_MAX_TURNS,
    MAX_TURNS,
    PuzzleEnv,
)
from puzzle3.solver import exact_distance
from rl_training.async_worker import PuzzleAsyncRolloutWorker

DEFAULT_CONFIG = Path("configs/rl.toml")


def read_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    bootstrap_args, _ = bootstrap.parse_known_args()
    defaults = read_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=bootstrap_args.config)
    parser.add_argument("--model", default=defaults.get("model", "Qwen/Qwen3.5-4B"))
    parser.add_argument(
        "--vllm-model", default=defaults.get("vllm_model", defaults.get("model"))
    )
    parser.add_argument(
        "--dataset", default=defaults.get("dataset", "data/eval_puzzles_31.jsonl")
    )
    parser.add_argument("--dataset-config", default=defaults.get("dataset_config"))
    parser.add_argument(
        "--dataset-split", default=defaults.get("dataset_split", "train")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(defaults.get("output_dir", "outputs/rl")),
    )
    parser.add_argument(
        "--vllm-server-url",
        default=defaults.get("vllm_server_url", "http://localhost:8000"),
    )
    parser.add_argument("--max-steps", type=int, default=defaults.get("max_steps", 100))
    parser.add_argument(
        "--save-steps", type=int, default=defaults.get("save_steps", 20)
    )
    parser.add_argument(
        "--logging-steps", type=int, default=defaults.get("logging_steps", 1)
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=defaults.get("per_device_train_batch_size", 1),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=defaults.get("gradient_accumulation_steps", 8),
    )
    parser.add_argument(
        "--num-generations", type=int, default=defaults.get("num_generations", 8)
    )
    parser.add_argument(
        "--max-turn-tokens", type=int, default=defaults.get("max_turn_tokens", 512)
    )
    parser.add_argument(
        "--max-turns", type=int, default=defaults.get("max_turns", DEFAULT_MAX_TURNS)
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=defaults.get("history_turns", DEFAULT_HISTORY_TURNS),
    )
    parser.add_argument(
        "--max-inflight-tasks", type=int, default=defaults.get("max_inflight_tasks", 16)
    )
    parser.add_argument(
        "--max-staleness", type=int, default=defaults.get("max_staleness", 1)
    )
    parser.add_argument(
        "--weight-sync-steps", type=int, default=defaults.get("weight_sync_steps", 1)
    )
    parser.add_argument(
        "--learning-rate", type=float, default=defaults.get("learning_rate", 1e-6)
    )
    parser.add_argument(
        "--temperature", type=float, default=defaults.get("temperature", 1.0)
    )
    parser.add_argument("--top-p", type=float, default=defaults.get("top_p", 1.0))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--report-to", default=defaults.get("report_to", "none"))
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("bf16", True),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("gradient_checkpointing", True),
    )
    return parser.parse_args()


def load_training_data(
    source: str, config: str | None, split: str, max_turns: int
) -> Dataset:
    path = Path(source)
    if path.is_file():
        dataset = load_dataset("json", data_files={split: str(path)}, split=split)
    elif config:
        dataset = load_dataset(source, config, split=split)
    else:
        dataset = load_dataset(source, split=split)

    if not dataset:
        raise ValueError("RL dataset is empty")

    def normalize(row: dict[str, Any]) -> dict[str, Any]:
        board = tuple(int(value) for value in row["board"])
        optimal_length = row["optimal_length"]
        if exact_distance(board) != optimal_length:
            raise ValueError("optimal_length does not match board distance")
        return {
            "board": list(board),
            "optimal_length": optimal_length,
            "max_turns": max_turns,
            "prompt": [{"role": "user", "content": ""}],
        }

    return dataset.map(normalize, desc="Validating puzzle RL rows")


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.num_generations < 2:
        raise ValueError("--num-generations must be at least 2 for GRPO")
    if not 1 <= args.max_turns <= MAX_TURNS:
        raise ValueError(f"--max-turns must be between 1 and {MAX_TURNS}")
    if args.history_turns < 0:
        raise ValueError("--history-turns must be non-negative")
    if args.max_turn_tokens <= 0:
        raise ValueError("--max-turn-tokens must be positive")

    dataset = load_training_data(
        args.dataset, args.dataset_config, args.dataset_split, args.max_turns
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = AsyncGRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_turn_tokens,
        max_tool_calling_iterations=args.max_turns,
        max_staleness=args.max_staleness,
        max_inflight_tasks=args.max_inflight_tasks,
        vllm_server_base_url=args.vllm_server_url,
        weight_sync_steps=args.weight_sync_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to=args.report_to,
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    worker = PuzzleAsyncRolloutWorker(
        model_name=args.vllm_model,
        dataset=dataset,
        reward_funcs=[],
        processing_class=tokenizer,
        tools=[],
        environment_factory=PuzzleEnv,
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight_tasks,
        vllm_server_url=args.vllm_server_url,
        max_tokens=args.max_turn_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=0,
        min_p=None,
        repetition_penalty=1.0,
        request_timeout=training_args.vllm_server_timeout,
        chat_template_kwargs=training_args.chat_template_kwargs,
        max_tool_calling_iterations=args.max_turns,
        log_completions=training_args.log_completions,
        num_completions_to_print=training_args.num_completions_to_print,
        fork_threshold_tokens=training_args.fork_threshold_tokens,
        history_turns=args.history_turns,
    )
    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_worker=worker,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
