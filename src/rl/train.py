"""GRPO training entrypoint."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any

import yaml
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from env.rewards import METRIC_KEYS, flat_metrics
from rl.curriculum import BUCKET_ORDER, TaskSampler
from rl.dataset import build_train_dataset_from_rows, load_rl_dataset, rows_from_dataset
from rl.rollout import build_rollout_func

TRL_KEYS = (
    "output_dir",
    "max_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "num_generations",
    "learning_rate",
    "bf16",
    "gradient_checkpointing",
    "max_completion_length",
    "use_vllm",
    "vllm_mode",
    "vllm_gpu_memory_utilization",
    "vllm_max_model_length",
    "logging_steps",
    "save_strategy",
    "save_steps",
    "report_to",
    "run_name",
    "seed",
    "log_completions",
    "num_completions_to_print",
    "resume_from_checkpoint",
    "gradient_checkpointing_kwargs",
    "warmup_ratio",
    "lr_scheduler_type",
    "optim",
)


@dataclass
class DatasetConfig:
    name: str = "saad1926q/15-puzzle"
    subset: str = "rl"
    split: str = "rl"


@dataclass
class CurriculumConfig:
    sigma: float = 0.75
    beta: float = 0.25
    min_prob: bool | float = 0.0
    scheduler_args: dict[str, Any] = field(default_factory=dict)

    def sampler_params(self) -> dict[str, Any]:
        args = {"mu_exp": self.beta, "sigma": self.sigma, "min_prob": self.min_prob, **self.scheduler_args}
        return {"curriculum_schedule": "gaussian", "bucket_order": BUCKET_ORDER, "scheduler_args": args}


@dataclass
class EnvConfig:
    max_turns: int = 120


@dataclass
class TrainConfig:
    model: str
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    trl: dict[str, Any] = field(default_factory=dict)


def load_config(path: str) -> TrainConfig:
    """Load a YAML training config."""

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    trl_raw = raw.get("trl", {}) or {}
    unknown = [key for key in trl_raw if key not in TRL_KEYS]
    if unknown:
        raise ValueError(f"unknown trl config keys: {sorted(unknown)}")

    trl_kwargs = {key: value for key, value in trl_raw.items() if key in TRL_KEYS}
    if trl_kwargs.get("resume_from_checkpoint") is None:
        trl_kwargs.pop("resume_from_checkpoint", None)

    return TrainConfig(
        model=raw["model"],
        dataset=DatasetConfig(**raw.get("dataset", {})),
        curriculum=CurriculumConfig(**raw.get("curriculum", {})),
        env=EnvConfig(**raw.get("env", {})),
        trl=trl_kwargs,
    )


def build_train_dataset(config: TrainConfig) -> Dataset:
    """Load the full RL split."""

    dataset = load_rl_dataset(config.dataset.name, config.dataset.subset, config.dataset.split)
    return build_train_dataset_from_rows(rows_from_dataset(dataset))


class CurriculumGRPOTrainer(GRPOTrainer):
    """GRPO trainer with curriculum sampling."""

    def __init__(self, scheduler_params: dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler_params = scheduler_params

    def _get_train_sampler(self, dataset: Dataset | None = None):
        dataset = dataset or self.train_dataset
        return TaskSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            seed=self.args.seed,
            total_iterations=self.args.max_steps,
            scheduler_params=self.scheduler_params,
        )


def make_reward_func():
    """Forward rollout rewards into GRPO."""

    def puzzle_reward(env_reward=None, verify_metrics=None, log_metric=None, **kwargs):
        if env_reward is None:
            raise ValueError("puzzle_reward requires env_reward")

        rewards = [float(reward) for reward in env_reward]
        metrics_list = verify_metrics or [{}] * len(rewards)

        if log_metric is not None:
            for item in metrics_list:
                for key, value in flat_metrics(item).items():
                    if key in METRIC_KEYS or key in ("turns", "reward"):
                        log_metric(f"puzzle/{key}", value)

        return rewards

    return puzzle_reward


def make_grpo_config(trl_kwargs: dict[str, Any]) -> GRPOConfig:
    return GRPOConfig(**trl_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a configs/trl/*.yaml file")
    parser.add_argument("--output-dir", default=None, help="override trl.output_dir")
    parser.add_argument("--resume", default=None, help="resume from a TRL checkpoint dir")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir is not None:
        config.trl["output_dir"] = args.output_dir
    if args.resume is not None:
        config.trl["resume_from_checkpoint"] = args.resume

    train_dataset = build_train_dataset(config)
    rollout_func = build_rollout_func(max_turns=config.env.max_turns)

    resume_from_checkpoint = config.trl.pop("resume_from_checkpoint", None)
    grpo_config = make_grpo_config(config.trl)

    trainer = CurriculumGRPOTrainer(
        scheduler_params=config.curriculum.sampler_params(),
        model=config.model,
        reward_funcs=[make_reward_func()],
        train_dataset=train_dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    if grpo_config.report_to and "wandb" in grpo_config.report_to:
        try:
            import wandb

            if wandb.run is not None:
                wandb.config.update(
                    {
                        "curriculum_schedule": "gaussian",
                        "sigma": config.curriculum.sigma,
                        "beta": config.curriculum.beta,
                        "min_prob": config.curriculum.min_prob,
                        "max_turns": config.env.max_turns,
                    }
                )
        except Exception:
            pass

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if grpo_config.output_dir:
        trainer.save_model(os.path.join(grpo_config.output_dir, "final"))


if __name__ == "__main__":
    main()
