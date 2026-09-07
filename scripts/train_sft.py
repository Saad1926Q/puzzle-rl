"""Train a Qwen causal LM on the replay-verified SFT decision dataset."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from sft_generation.training import (
    is_decision_record,
    normalize_tool_arguments,
    validate_decision_record,
)

DEFAULT_CONFIG = Path("configs/sft.toml")


def _config_values(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        values = tomllib.load(file)
    return values


def parse_args() -> argparse.Namespace:
    """Parse CLI overrides on top of the TOML configuration."""
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    bootstrap_args, _ = bootstrap.parse_known_args()
    defaults = _config_values(bootstrap_args.config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=bootstrap_args.config)
    parser.add_argument("--model", default=defaults.get("model", "Qwen/Qwen3.5-4B"))
    parser.add_argument("--dataset", default=defaults.get("dataset", "saad1926q/8-puzzle"))
    parser.add_argument("--dataset-config", default=defaults.get("dataset_config", "sft"))
    parser.add_argument("--dataset-split", default=defaults.get("dataset_split", "train"))
    parser.add_argument("--output-dir", type=Path, default=Path(defaults.get("output_dir", "outputs/sft")))
    parser.add_argument("--max-steps", type=int, default=defaults.get("max_steps", 120))
    parser.add_argument("--save-steps", type=int, default=defaults.get("save_steps", 10))
    parser.add_argument("--save-total-limit", type=int, default=defaults.get("save_total_limit", 6))
    parser.add_argument("--logging-steps", type=int, default=defaults.get("logging_steps", 1))
    parser.add_argument("--per-device-train-batch-size", type=int, default=defaults.get("per_device_train_batch_size", 1))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults.get("gradient_accumulation_steps", 16))
    parser.add_argument("--learning-rate", type=float, default=defaults.get("learning_rate", 5e-5))
    parser.add_argument("--warmup-ratio", type=float, default=defaults.get("warmup_ratio", 0.05))
    parser.add_argument("--max-length", type=int, default=defaults.get("max_length", 2048))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--lora-r", type=int, default=defaults.get("lora_r", 16))
    parser.add_argument("--lora-alpha", type=int, default=defaults.get("lora_alpha", 32))
    parser.add_argument("--lora-dropout", type=float, default=defaults.get("lora_dropout", 0.05))
    parser.add_argument("--report-to", default=defaults.get("report_to", "none"))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=defaults.get("bf16", True))
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("gradient_checkpointing", True),
    )
    return parser.parse_args()


def load_decision_dataset(args: argparse.Namespace) -> Dataset:
    """Load and normalize supervised decision rows from the SFT configuration."""
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    decision_rows = dataset.filter(is_decision_record)
    if len(decision_rows) == 0:
        raise ValueError("SFT training dataset contains no decision rows")
    decision_rows = decision_rows.map(normalize_tool_arguments)
    for row in decision_rows.select(range(min(8, len(decision_rows)))):
        validate_decision_record(row)
    return decision_rows


def main() -> None:
    """Run step-bounded LoRA SFT."""
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.bf16 and not torch.cuda.is_available():
        raise ValueError("--bf16 requires a CUDA device; pass --no-bf16 on CPU")

    dataset = load_decision_dataset(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else None
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        eos_token=tokenizer.eos_token or "<|im_end|>",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
