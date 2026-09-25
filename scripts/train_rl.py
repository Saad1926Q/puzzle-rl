"""Train the 8-puzzle policy with synchronous bounded-history GRPO."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig

from puzzle3.environment import (
    DEFAULT_HISTORY_TURNS,
    DEFAULT_MAX_TURNS,
    MAX_TURNS,
    PuzzleEnv,
)
from puzzle3.solver import exact_distance
from rl_training.grpo_trainer import PuzzleGRPOTrainer

QWEN35_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


DEFAULT_CONFIG = Path("configs/rl/run_1.toml")


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
        "--dataset",
        default=defaults.get("dataset", "saad1926q/8-puzzle"),
    )
    parser.add_argument(
        "--dataset-subset",
        default=defaults.get("dataset_subset", "rl"),
    )
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
    parser.add_argument(
        "--vllm-mode",
        choices=("colocate", "server"),
        default=defaults.get("vllm_mode", "colocate"),
    )
    parser.add_argument(
        "--use-vllm",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("use_vllm", True),
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=defaults.get("vllm_gpu_memory_utilization", 0.3),
    )
    parser.add_argument(
        "--vllm-enable-sleep-mode",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("vllm_enable_sleep_mode", True),
    )
    parser.add_argument("--max-steps", type=int, default=defaults.get("max_steps", 100))
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=defaults.get("num_train_epochs", 1.0),
    )
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
    parser.add_argument(
        "--use-lora",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("use_lora", True),
    )
    parser.add_argument("--lora-r", type=int, default=defaults.get("lora_r", 16))
    parser.add_argument(
        "--lora-alpha", type=int, default=defaults.get("lora_alpha", 16)
    )
    return parser.parse_args()


def load_training_data(
    dataset_name: str, subset: str, split: str, max_turns: int
) -> Dataset:
    dataset = load_dataset(dataset_name, name=subset, split=split)

    if len(dataset) == 0:
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
    if not args.use_vllm:
        raise ValueError("synchronous puzzle training requires --use-vllm")
    if not 0.0 < args.vllm_gpu_memory_utilization < 1.0:
        raise ValueError("--vllm-gpu-memory-utilization must be between 0 and 1")
    if args.max_steps == 0:
        raise ValueError("--max-steps must be positive or -1 for epoch-driven training")
    if args.max_steps < 0 and args.num_train_epochs <= 0:
        raise ValueError("--num-train-epochs must be positive when --max-steps is -1")
    if args.num_generations < 2:
        raise ValueError("--num-generations must be at least 2 for GRPO")
    if not 1 <= args.max_turns <= MAX_TURNS:
        raise ValueError(f"--max-turns must be between 1 and {MAX_TURNS}")
    if args.history_turns < 0:
        raise ValueError("--history-turns must be non-negative")
    if args.max_turn_tokens <= 0:
        raise ValueError("--max-turn-tokens must be positive")
    if args.use_lora and args.lora_r <= 0:
        raise ValueError("--lora-r must be positive")
    if args.use_lora and args.lora_alpha <= 0:
        raise ValueError("--lora-alpha must be positive")

    dataset = load_training_data(
        args.dataset, args.dataset_subset, args.dataset_split, args.max_turns
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        generation_batch_size=args.num_generations,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_turn_tokens,
        max_tool_calling_iterations=args.max_turns,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        vllm_importance_sampling_correction=True,
        loss_type="dapo",
        epsilon=0.2,
        epsilon_high=0.28,
        mask_truncated_completions=True,
        beta=0.0,
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
        chat_template_kwargs={"enable_thinking": False},
    )
    trainer_kwargs: dict[str, Any] = {
        "model": args.model,
        "args": training_args,
        "train_dataset": dataset,
        "processing_class": tokenizer,
        "environment_factory": PuzzleEnv,
        "history_turns": args.history_turns,
    }
    if args.use_lora:
        trainer_kwargs["peft_config"] = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            use_rslora=False,
            target_modules=QWEN35_LORA_TARGET_MODULES,
        )
    trainer = PuzzleGRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))



if __name__ == "__main__":
    main()
