"""E2H-style curriculum sampling over string difficulty buckets.

Load the full dataset once; ``TaskSampler`` changes bucket probabilities every
optimizer iteration.
"""

from __future__ import annotations

import math
from typing import Any

BUCKET_ORDER = ("trivial", "easy", "medium", "hard")

DEFAULT_SIGMA = 0.75
DEFAULT_BETA = 0.25


class TaskSampler:
    """E2H-style curriculum sampler over the dataset's string ``bucket`` column."""

    def __init__(
        self,
        data_source,
        mini_repeat_count: int,
        batch_size: int,
        repeat_count: int,
        seed: int,
        total_iterations: int,
        scheduler_params: dict[str, Any],
    ):
        import numpy as np

        self.dataset = data_source
        if "bucket" not in self.dataset.column_names:
            raise ValueError("curriculum sampler requires a 'bucket' column")

        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.rng = np.random.default_rng(int(seed))
        self.total_iterations = total_iterations
        self.bucket_order = tuple(scheduler_params.get("bucket_order", BUCKET_ORDER))
        self.num_tasks = len(self.bucket_order)

        schedule = scheduler_params.get("curriculum_schedule", scheduler_params.get("schedule", "gaussian"))
        if schedule != "gaussian":
            raise ValueError(f"unsupported curriculum schedule: {schedule}")

        self.scheduler_args = dict(scheduler_params.get("scheduler_args", {}))
        if "mu_exp" not in self.scheduler_args and "beta" in scheduler_params:
            self.scheduler_args["mu_exp"] = scheduler_params["beta"]
        if "sigma" not in self.scheduler_args and "sigma" in scheduler_params:
            self.scheduler_args["sigma"] = scheduler_params["sigma"]
        self.scheduler_args.setdefault("mu_exp", DEFAULT_BETA)
        self.scheduler_args.setdefault("sigma", DEFAULT_SIGMA)
        self.scheduler_args.setdefault("min_prob", 0.0)

        bucket_col = np.asarray(self.dataset["bucket"])
        self.indices_by_bucket = {}
        for bucket in self.bucket_order:
            indices = np.where(bucket_col == bucket)[0]
            if len(indices) == 0:
                raise ValueError(f"no rows found for bucket: {bucket}")
            self.indices_by_bucket[bucket] = self.rng.permutation(indices)

    @staticmethod
    def _gaussian_schedule(
        t: int,
        T: int,
        num_tasks: int,
        mu_exp: float,
        sigma: float,
        min_prob: bool | float = 0.0,
        **kwargs,
    ) -> list[float]:
        if T <= 0:
            raise ValueError("total_iterations must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        mu = (t / T) ** mu_exp * (num_tasks - 1)
        if min_prob is True:
            p_min = 2 / (num_tasks * (num_tasks + 1))
        elif min_prob is False:
            p_min = 0.0
        else:
            p_min = float(min_prob)
        if num_tasks * p_min > 1:
            raise ValueError("num_tasks * min_prob must not exceed 1")

        base = [math.exp(-((i - mu) ** 2) / (2 * sigma**2)) for i in range(num_tasks)]
        total = sum(base)
        return [p_min + (1 - num_tasks * p_min) * (b / total) for b in base]

    def __len__(self) -> int:
        return self.total_iterations * self.batch_size * self.mini_repeat_count * self.repeat_count

    def __iter__(self):
        import numpy as np

        ptrs = {bucket: 0 for bucket in self.bucket_order}
        bucket_ids = np.arange(self.num_tasks)

        for i in range(self.total_iterations):
            probs = np.asarray(
                self._gaussian_schedule(i, self.total_iterations, self.num_tasks, **self.scheduler_args),
                dtype=float,
            )
            chosen = self.rng.choice(bucket_ids, size=self.batch_size, p=probs, replace=True)
            batch_indices = []

            for bucket_id in chosen:
                bucket = self.bucket_order[int(bucket_id)]
                indices = self.indices_by_bucket[bucket]
                p = ptrs[bucket] % len(indices)
                batch_indices.append(int(indices[p]))
                ptrs[bucket] += 1

            self.last_batch_indices = batch_indices
            self.last_chosen_buckets = [self.bucket_order[int(i)] for i in chosen]

            for _ in range(self.repeat_count):
                for index in batch_indices:
                    for _ in range(self.mini_repeat_count):
                        yield index

