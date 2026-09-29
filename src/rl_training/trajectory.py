"""Human-readable append-only trajectory artifacts for RL training."""

from __future__ import annotations

import json
import os
from pathlib import Path

from rl_training.sync_rollout import EpisodeRollout


class RLTrajectoryWriter:
    """Write one JSON record per completed GRPO generation group."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("x", encoding="utf-8")

    def write_group(
        self,
        training_step: int,
        group: int,
        episodes: list[EpisodeRollout],
        advantages: list[float],
    ) -> None:
        if len(episodes) != len(advantages):
            raise ValueError("trajectory episodes and advantages must have equal lengths")
        record = {
            "training_step": training_step,
            "group": group,
            "attempts": [
                episode.to_dict(advantage=advantage)
                for episode, advantage in zip(episodes, advantages, strict=True)
            ],
        }
        self._file.write(
            json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
        )
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "RLTrajectoryWriter":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()
