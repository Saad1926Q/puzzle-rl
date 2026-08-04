"""Evaluate puzzle models with the same environment used for GRPO."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable

from tqdm import tqdm

from env.protocol import build_system_prompt
from env.session import FifteenPuzzleSession

EVAL_DATASET_NAME = "saad1926q/15-puzzle"
EVAL_DATASET_SUBSET = "eval"
EVAL_DATASET_SPLIT = "eval"


@dataclass
class EvalConfig:
    output_path: str
    model: str
    max_turns: int = 120
    num_generations: int = 1
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    batch_size: int = 1
    keep_history: bool = False
    save_transcript: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None


@dataclass
class EvalJob:
    session: FifteenPuzzleSession
    puzzle_index: int
    generation_index: int
    transcript: list[dict[str, str]]
    responses: list[str]
    response_tokens: list[int]

    @property
    def key(self) -> tuple[int, int]:
        return self.puzzle_index, self.generation_index


def load_eval_rows() -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        EVAL_DATASET_NAME,
        EVAL_DATASET_SUBSET,
        split=EVAL_DATASET_SPLIT,
    )
    return [dict(row) for row in dataset]


def write_jsonl_record(path: str, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_hf_model(model_name_or_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"device_map": "auto"}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.eval()
    return tokenizer, model


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return model.device


def _trim_generated_tokens(token_ids, tokenizer, model) -> list[int]:
    raw_ids = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
    ids = [int(token_id) for token_id in raw_ids]
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    eos_ids.discard(None)

    for index, token_id in enumerate(ids):
        if token_id in eos_ids:
            return ids[: index + 1]

    pad_token_id = tokenizer.pad_token_id
    while ids and pad_token_id is not None and ids[-1] == pad_token_id:
        ids.pop()
    return ids


def call_model_batch(
    messages_batch: list[list[dict[str, str]]],
    tokenizer,
    model,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
) -> list[tuple[str, int]]:
    """Generate one response for each active session."""

    import torch

    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_batch
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(_model_device(model))

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
        if top_k is not None:
            generation_kwargs["top_k"] = top_k

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    prompt_width = inputs["input_ids"].shape[1]
    results: list[tuple[str, int]] = []
    for output in outputs:
        generated_tokens = _trim_generated_tokens(output[prompt_width:], tokenizer, model)
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        results.append((text, len(generated_tokens)))
    return results


def call_model(
    messages: list[dict[str, str]],
    tokenizer,
    model,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
) -> tuple[str, int]:
    """Generate one response through the batched generation path."""

    return call_model_batch(
        [messages],
        tokenizer,
        model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )[0]


def build_eval_messages(
    session: FifteenPuzzleSession,
    transcript: list[dict[str, str]],
    *,
    keep_history: bool,
) -> list[dict[str, str]]:
    system = {"role": "system", "content": build_system_prompt()}
    user = {"role": "user", "content": session.current_board_prompt()}

    if keep_history:
        return [system, *transcript, user]
    return [system, user]


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if "initial_board" not in out and "board" in out:
        out["initial_board"] = out["board"]
    if "board" not in out and "initial_board" in out:
        out["board"] = out["initial_board"]
    return out


def create_eval_job(
    candidate: dict[str, Any],
    config: EvalConfig,
    *,
    puzzle_index: int,
    generation_index: int,
) -> EvalJob:
    row = normalize_row(candidate)
    return EvalJob(
        session=FifteenPuzzleSession.from_row(row, max_turns=config.max_turns),
        puzzle_index=puzzle_index,
        generation_index=generation_index,
        transcript=[],
        responses=[],
        response_tokens=[],
    )


def advance_jobs(
    jobs: list[EvalJob],
    tokenizer,
    model,
    config: EvalConfig,
) -> None:
    """Advance each active trajectory by one model turn."""

    messages_batch = [
        build_eval_messages(job.session, job.transcript, keep_history=config.keep_history)
        for job in jobs
    ]
    outputs = call_model_batch(
        messages_batch,
        tokenizer,
        model,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
    )
    if len(outputs) != len(jobs):
        raise RuntimeError(f"model returned {len(outputs)} responses for {len(jobs)} jobs")

    for job, (response_text, token_count) in zip(jobs, outputs, strict=True):
        board_prompt = job.session.current_board_prompt()
        job.responses.append(response_text)
        job.response_tokens.append(token_count)
        job.transcript.append({"role": "user", "content": board_prompt})
        job.transcript.append({"role": "assistant", "content": response_text})
        job.session.apply_model_response(response_text)


def finalize_job(job: EvalJob, config: EvalConfig) -> dict[str, Any]:
    session = job.session
    metric = session.metrics()
    result: dict[str, Any] = {
        "puzzle_index": job.puzzle_index,
        "generation_index": job.generation_index,
        "model": config.model,
        "bucket": session.bucket,
        "board": list(session.initial_board),
        "scramble_depth": session.scramble_depth,
        "optimal_moves": list(session.optimal_moves or []),
        "optimal_length": session.optimal_length,
        "moves": list(session.moves_taken),
        "move_count": session.moves_taken_count,
        "terminal_reason": session.terminal_reason,
        "solved": session.solved,
        "illegal_move": session.illegal_move,
        "format_failure": session.format_failure,
        "reward": session.final_reward(),
        "efficiency": metric["efficiency"],
        "response_tokens": list(job.response_tokens),
        "total_response_tokens": sum(job.response_tokens),
        "last_response": job.responses[-1] if job.responses else None,
    }
    if config.save_transcript:
        result["transcript"] = list(job.transcript)
        result["responses"] = list(job.responses)
    return result


def run_batched_trajectories(
    candidates: list[dict[str, Any]],
    tokenizer,
    model,
    config: EvalConfig,
    progress,
) -> list[dict[str, Any]]:
    pending = deque(
        create_eval_job(
            candidate,
            config,
            puzzle_index=puzzle_index,
            generation_index=generation_index,
        )
        for puzzle_index, candidate in enumerate(candidates)
        for generation_index in range(config.num_generations)
    )
    active: list[EvalJob] = []
    completed: dict[tuple[int, int], dict[str, Any]] = {}

    while pending or active:
        while pending and len(active) < config.batch_size:
            active.append(pending.popleft())

        advance_jobs(active, tokenizer, model, config)

        continuing: list[EvalJob] = []
        for job in active:
            if job.session.done:
                completed[job.key] = finalize_job(job, config)
                progress.update(1)
            else:
                continuing.append(job)
        active = continuing

    return [
        completed[(puzzle_index, generation_index)]
        for puzzle_index in range(len(candidates))
        for generation_index in range(config.num_generations)
    ]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_puzzle: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_bucket[str(result["bucket"])].append(result)
        by_puzzle[int(result["puzzle_index"])].append(result)

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        terminal_counts = Counter(str(item["terminal_reason"]) for item in group)
        return {
            "n": len(group),
            "reward_mean": _mean(float(item["reward"]) for item in group),
            "solved_rate": _mean(float(item["solved"]) for item in group),
            "illegal_move_rate": _mean(float(item["illegal_move"]) for item in group),
            "format_failure_rate": _mean(float(item["format_failure"]) for item in group),
            "max_turns_rate": _mean(float(item["terminal_reason"] == "max_turns") for item in group),
            "turns_mean": _mean(float(item["move_count"]) for item in group),
            "efficiency_mean": _mean(float(item["efficiency"]) for item in group),
            "tokens_mean": _mean(float(item["total_response_tokens"]) for item in group),
            "terminal_reason_counts": dict(terminal_counts),
        }

    pass_at_n = _mean(any(item["solved"] for item in group) for group in by_puzzle.values())
    summary = {
        "overall": summarize_group(results),
        "pass_at_num_generations": pass_at_n,
        "num_puzzles": len(by_puzzle),
        "num_rollouts": len(results),
        "by_bucket": {bucket: summarize_group(group) for bucket, group in sorted(by_bucket.items())},
    }
    return summary


def log_summary_to_wandb(summary: dict[str, Any], config: EvalConfig) -> None:
    if config.wandb_project is None:
        return

    import wandb

    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        job_type="eval",
        config={
            "model": config.model,
            "dataset": EVAL_DATASET_NAME,
            "dataset_subset": EVAL_DATASET_SUBSET,
            "dataset_split": EVAL_DATASET_SPLIT,
            "max_turns": config.max_turns,
            "num_generations": config.num_generations,
            "batch_size": config.batch_size,
            "keep_history": config.keep_history,
            "temperature": config.temperature,
        },
    )
    metrics: dict[str, float] = {}
    for key, value in summary["overall"].items():
        if isinstance(value, (int, float)):
            metrics[f"eval/overall/{key}"] = float(value)
    metrics["eval/pass_at_num_generations"] = float(summary["pass_at_num_generations"])
    for bucket, bucket_summary in summary["by_bucket"].items():
        for key, value in bucket_summary.items():
            if isinstance(value, (int, float)):
                metrics[f"eval/{bucket}/{key}"] = float(value)
    wandb.log(metrics)
    run.finish()


def run_eval(config: EvalConfig) -> dict[str, Any]:
    if config.max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if config.num_generations <= 0:
        raise ValueError("num_generations must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    candidates = load_eval_rows()
    tokenizer, model = load_hf_model(config.model)

    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)
    with open(config.output_path, "w", encoding="utf-8"):
        pass

    total = len(candidates) * config.num_generations
    progress = tqdm(total=total, desc="Running eval", unit="rollout")
    results = run_batched_trajectories(candidates, tokenizer, model, config, progress)
    progress.close()

    for result in results:
        write_jsonl_record(config.output_path, result)

    summary = summarize_results(results)
    summary_path = os.path.splitext(config.output_path)[0] + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    log_summary_to_wandb(summary, config)
    print(json.dumps(summary, indent=2))
    print(f"Wrote raw results to {config.output_path}")
    print(f"Wrote summary to {summary_path}")
    return summary


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", dest="output_path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--keep-history", action="store_true")
    parser.add_argument("--save-transcript", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()
    return EvalConfig(**vars(args))


def main() -> None:
    run_eval(parse_args())


if __name__ == "__main__":
    main()
