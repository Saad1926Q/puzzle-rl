"""Preflight SFT chat-template rendering and completion-only loss boundaries."""

from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from sft_generation.training import render_training_record, validate_decision_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="saad1926q/8-puzzle")
    parser.add_argument("--dataset-config", default="sft")
    parser.add_argument("--split", default="train")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 4, 1000])
    parser.add_argument("--max-length", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    decision_rows = dataset.filter(
        lambda row: row.get("metadata", {}).get("record_type") == "decision"
    )
    if not decision_rows:
        raise ValueError("no decision rows found for SFT preflight")

    for index in args.indices:
        if index < 0 or index >= len(decision_rows):
            raise ValueError(f"index {index} is outside the decision dataset")
        record = decision_rows[index]
        validate_decision_record(record)
        lengths = render_training_record(tokenizer, record)
        if lengths["total_tokens"] > args.max_length:
            raise ValueError(
                f"row {index} has {lengths['total_tokens']} tokens, "
                f"over max_length={args.max_length}"
            )
        print(index, lengths)

    print(f"preflight passed for {len(args.indices)} rows")


if __name__ == "__main__":
    main()
