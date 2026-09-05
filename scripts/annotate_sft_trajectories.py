"""Annotate verified trajectories and build the final SFT dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.constants import DEFAULT_OPENROUTER_BASE_URL
from evaluation.protocol import DEFAULT_API_KEY_ENV, get_api_key
from sft_generation.annotation import AnnotationConfig, annotation_futures
from sft_generation.constants import DEFAULT_TEACHER_MODEL
from sft_generation.formatting import build_sft_record
from sft_generation.storage import append_jsonl, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse annotation options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_OPENROUTER_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--upstream-provider", action="append", default=[])
    parser.add_argument("--allow-fallbacks", action="store_true")
    parser.add_argument("--allow-parameters", action="store_true")
    parser.add_argument("--data-collection", choices=("allow", "deny"), default="deny")
    parser.add_argument("--provider-retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--annotations-output", type=Path)
    parser.add_argument("--dataset-output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Annotate missing moves and rebuild complete SFT records."""

    args = parse_args()
    trajectories = list(read_jsonl(args.input))
    output_dir = args.input.parent
    annotations_path = args.annotations_output or output_dir / "annotations.jsonl"
    dataset_path = args.dataset_output or output_dir / "sft_dataset.jsonl"
    stored = {
        (record.get("source_id"), record.get("turn")): record
        for record in read_jsonl(annotations_path)
        if record.get("valid")
    }
    missing = {
        (trajectory["source_id"], step["turn"])
        for trajectory in trajectories
        for step in trajectory["steps"]
        if (trajectory["source_id"], step["turn"]) not in stored
    }
    config = AnnotationConfig(
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        thinking=not args.no_thinking,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        top_p=args.top_p,
        upstream_providers=tuple(args.upstream_provider),
        allow_fallbacks=args.allow_fallbacks,
        require_parameters=not args.allow_parameters,
        data_collection=args.data_collection,
        provider_retries=args.provider_retries,
        retry_delay=args.retry_delay,
    )
    if missing:
        stored.update(
            annotation_futures(
                trajectories,
                api_key=get_api_key(args.api_key_env),
                config=config,
                parallelism=args.parallelism,
                skip_keys=set(stored),
                on_result=lambda record: append_jsonl(annotations_path, record),
            )
        )
    ordered_annotations = [stored[key] for key in sorted(stored)]
    write_jsonl(annotations_path, ordered_annotations)
    records = []
    for trajectory in sorted(trajectories, key=lambda item: item["source_id"]):
        annotations = [
            stored[(trajectory["source_id"], step["turn"])]
            for step in trajectory["steps"]
            if (trajectory["source_id"], step["turn"]) in stored
        ]
        if len(annotations) != len(trajectory["steps"]) or not all(
            item.get("valid") for item in annotations
        ):
            continue
        records.append(build_sft_record(trajectory, annotations))
    write_jsonl(dataset_path, records)
    write_json(
        output_dir / "annotation_metadata.json",
        {
            "input": str(args.input),
            "model": args.model,
            "parallelism": args.parallelism,
            "max_tokens": args.max_tokens,
            "thinking": not args.no_thinking,
            "reasoning_effort": args.reasoning_effort,
            "annotation_count": len(ordered_annotations),
            "dataset_count": len(records),
        },
    )
    print(f"annotations={len(ordered_annotations)} datasets={len(records)}")


if __name__ == "__main__":
    main()
