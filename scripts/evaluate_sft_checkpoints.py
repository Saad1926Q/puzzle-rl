"""Evaluate saved SFT checkpoints on the fresh puzzle validation split."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.constants import DEFAULT_MAX_TURNS, SLIDE_TILE_TOOL
from evaluation.dataset import load_examples
from evaluation.evaluator import EvaluationResult, evaluate
from evaluation.protocol import (
    HistoryTurn,
    build_chat_completion_messages,
    parse_tile,
)


def checkpoint_paths(root: Path, requested: list[str] | None) -> list[Path]:
    """Resolve checkpoints in numeric step order."""
    if requested:
        paths = [root / value for value in requested]
    else:
        paths = sorted(
            root.glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
    if not paths:
        raise FileNotFoundError(f"no checkpoints found under {root}")
    missing = [path for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"checkpoint directories do not exist: {missing}")
    return paths


class LocalCheckpointAgent:
    """Generate puzzle tool calls directly from one local checkpoint."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        base_model: str,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.last_response_metadata: dict[str, Any] | None = None
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if (checkpoint / "adapter_config.json").exists():
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                checkpoint,
                dtype=dtype,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                dtype=dtype,
                device_map="auto",
            )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def next_action(
        self,
        board: tuple[int, ...],
        history: tuple[HistoryTurn, ...] = (),
        *,
        include_reasoning: bool = False,
    ) -> str:
        messages = build_chat_completion_messages(
            board,
            history,
            include_reasoning=include_reasoning,
        )
        for message in messages:
            for tool_call in message.get("tool_calls", []):
                function = tool_call.get("function", tool_call)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    function["arguments"] = json.loads(arguments)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tools=[SLIDE_TILE_TOOL],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, Mapping) or hasattr(encoded, "input_ids"):
            model_inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            input_length = model_inputs["input_ids"].shape[-1]
        else:
            model_inputs = {"input_ids": encoded.to(self.device)}
            input_length = model_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        response = self.tokenizer.decode(
            generated[0][input_length:],
            skip_special_tokens=False,
        )
        tile = parse_tile(response)
        if tile is None:
            match = re.search(r"<parameter=tile>\s*([1-8])\s*</parameter>", response)
            tile = int(match.group(1)) if match else None
        if tile is None:
            match = re.search(r"[\\\"']tile[\\\"']\s*:\s*([1-8])", response)
            tile = int(match.group(1)) if match else None
        self.last_response_metadata = {
            "status": "ok" if tile is not None else "invalid_response",
            "reasoning_content": response if include_reasoning else "",
        }
        return json.dumps({"tile": tile}) if tile is not None else response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dataset", default="saad1926q/8-puzzle")
    parser.add_argument("--config", default="sft")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--keep-history", action="store_true", default=True)
    parser.add_argument("--keep-reasoning", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        require_optimal_actions=False,
    )
    results: dict[str, Any] = {}
    for checkpoint in checkpoint_paths(args.checkpoint_root, args.checkpoints):
        agent = LocalCheckpointAgent(
            checkpoint,
            base_model=args.base_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        evaluation: EvaluationResult = evaluate(
            examples,
            agent=agent,
            max_turns=args.max_turns,
            num_rollouts=args.num_rollouts,
            keep_history=args.keep_history,
            keep_reasoning=args.keep_reasoning,
        )
        results[checkpoint.name] = evaluation.to_dict()
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for checkpoint, result in results.items():
        print(checkpoint, result["summary"])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
