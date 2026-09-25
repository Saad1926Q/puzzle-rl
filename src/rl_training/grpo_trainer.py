"""Synchronous GRPO adapter for bounded-history puzzle rollouts."""

from __future__ import annotations

from typing import Any

import torch
from trl import GRPOTrainer
from trl.trainer.utils import nanstd, pad, split_tensor_dict

from rl_training.sync_rollout import EpisodeRollout, TurnRecord, generate_episode_group


class PuzzleGRPOTrainer(GRPOTrainer):
    """Feed exact per-turn puzzle contexts into TRL's standard GRPO loss."""

    def __init__(self, *args: Any, history_turns: int, **kwargs: Any) -> None:
        if history_turns < 0:
            raise ValueError("history_turns must be non-negative")
        self.history_turns = history_turns
        super().__init__(*args, **kwargs)

    def _episode_advantages(
        self, episodes: list[EpisodeRollout], device: torch.device
    ) -> torch.Tensor:
        rewards = torch.tensor(
            [episode.reward for episode in episodes], dtype=torch.float32, device=device
        )
        mean = rewards.mean()
        std = nanstd(rewards)
        advantages = rewards - mean
        if self.scale_rewards != "none":
            advantages = advantages / (std + 1e-4)
        return advantages

    @staticmethod
    def _rows_from_episodes(
        episodes: list[EpisodeRollout], advantages: torch.Tensor
    ) -> list[tuple[TurnRecord, float, bool]]:
        rows: list[tuple[TurnRecord, float, bool]] = []
        for episode_index, episode in enumerate(episodes):
            advantage = float(advantages[episode_index].item())
            for turn in episode.turns:
                rows.append((turn, advantage, not episode.truncated))
        return rows

    def _pad_training_rows(
        self,
        rows: list[tuple[TurnRecord, float, bool]],
        *,
        mode: str,
    ) -> list[tuple[TurnRecord, float, bool]]:
        if not rows:
            raise RuntimeError("Puzzle rollouts produced no model-generated turns")
        if mode != "train":
            return rows
        chunk_count = self.args.steps_per_generation
        remainder = len(rows) % chunk_count
        if remainder == 0:
            return rows
        pad_count = chunk_count - remainder
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer must define pad_token_id")
        dummy = TurnRecord([pad_id], [pad_id], [0.0])
        rows.extend((dummy, 0.0, False) for _ in range(pad_count))
        return rows

    def _build_model_inputs(
        self,
        rows: list[tuple[TurnRecord, float, bool]],
        *,
        mode: str,
    ) -> dict[str, torch.Tensor]:
        device = self.accelerator.device
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer must define pad_token_id")

        prompt_ids_list: list[torch.Tensor] = []
        prompt_masks_list: list[torch.Tensor] = []
        completion_ids_list: list[torch.Tensor] = []
        completion_masks_list: list[torch.Tensor] = []
        sampling_logprobs_list: list[torch.Tensor] = []
        advantages: list[float] = []

        for turn, advantage, trainable in rows:
            prompt_ids = turn.prompt_ids or [pad_id]
            completion_ids = turn.completion_ids or [pad_id]
            completion_logprobs = turn.logprobs or [0.0]
            if len(completion_logprobs) != len(completion_ids):
                raise ValueError("generation logprobs must align with completion token IDs")
            prompt_ids_list.append(torch.tensor(prompt_ids, dtype=torch.long))
            prompt_masks_list.append(
                torch.ones(len(prompt_ids), dtype=torch.long)
            )
            completion_ids_list.append(torch.tensor(completion_ids, dtype=torch.long))
            completion_masks_list.append(
                torch.full(
                    (len(completion_ids),),
                    1 if trainable else 0,
                    dtype=torch.long,
                )
            )
            sampling_logprobs_list.append(
                torch.tensor(completion_logprobs, dtype=torch.float32)
            )
            advantages.append(advantage)

        prompt_ids = pad(
            prompt_ids_list,
            padding_value=pad_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)
        prompt_mask = pad(
            prompt_masks_list,
            padding_value=0,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)
        completion_ids = pad(
            completion_ids_list,
            padding_value=pad_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)
        completion_mask = pad(
            completion_masks_list,
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)
        sampling_logprobs = pad(
            sampling_logprobs_list,
            padding_value=0.0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        with torch.no_grad():
            old_logprobs, _, _ = self._get_per_token_logps_and_entropies(
                self.model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep,
                batch_size=self.args.per_device_train_batch_size,
            )

        output: dict[str, torch.Tensor] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
            "old_per_token_logps": old_logprobs.detach(),
            "num_items_in_batch": self.accelerator.gather(completion_mask.sum()).sum(),
        }

        if self.use_vllm and self.vllm_importance_sampling_correction:
            difference = (old_logprobs - sampling_logprobs) * completion_mask
            difference = torch.nan_to_num(difference, nan=0.0)
            mode_name = self.vllm_importance_sampling_mode
            if mode_name in {"sequence_mask", "sequence_truncate"}:
                difference = difference.sum(dim=-1, keepdim=True)
            ratio = torch.exp(difference)
            if mode_name in {"token_truncate", "sequence_truncate"}:
                ratio = torch.clamp(
                    ratio,
                    min=self.vllm_importance_sampling_clip_min,
                    max=self.vllm_importance_sampling_clip_max,
                )
            elif mode_name in {"token_mask", "sequence_mask"}:
                minimum = (
                    self.vllm_importance_sampling_clip_min
                    if self.vllm_importance_sampling_clip_min is not None
                    else -float("inf")
                )
                maximum = (
                    self.vllm_importance_sampling_clip_max
                    if self.vllm_importance_sampling_clip_max is not None
                    else float("inf")
                )
                invalid = (ratio < minimum) | (ratio > maximum)
                ratio = ratio.masked_fill(invalid, 0.0)
            else:
                raise ValueError(f"Unknown vLLM importance sampling mode: {mode_name}")
            output["sampling_per_token_logps"] = sampling_logprobs
            output["importance_sampling_ratio"] = ratio

        return output

    def _log_rollout_metrics(
        self, episodes: list[EpisodeRollout], advantages: torch.Tensor, mode: str
    ) -> None:
        rewards = [episode.reward for episode in episodes]
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        self._metrics[mode]["reward"].append(float(reward_tensor.mean()))
        self._metrics[mode]["reward_std"].append(float(nanstd(reward_tensor)))
        self._metrics[mode]["frac_reward_zero_std"].append(
            float(torch.isclose(nanstd(reward_tensor), torch.tensor(0.0)))
        )
        self._metrics[mode]["rollout/turns_mean"].append(
            sum(len(episode.turns) for episode in episodes) / len(episodes)
        )
        self._metrics[mode]["rollout/truncated_rate"].append(
            sum(episode.truncated for episode in episodes) / len(episodes)
        )
        self._metrics[mode]["rollout/solved_rate"].append(
            sum(episode.outcome == "solved" for episode in episodes) / len(episodes)
        )

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        if len(inputs) % self.num_generations != 0:
            raise ValueError(
                f"generation batch size {len(inputs)} must be divisible by num_generations "
                f"{self.num_generations}"
            )
        mode = "train" if self.model.training else "eval"
        all_rows: list[tuple[TurnRecord, float, bool]] = []
        all_episodes: list[EpisodeRollout] = []
        for start in range(0, len(inputs), self.num_generations):
            episodes = generate_episode_group(
                self,
                [dict(row) for row in inputs[start : start + self.num_generations]],
            )
            advantages = self._episode_advantages(episodes, self.accelerator.device)
            all_episodes.extend(episodes)
            all_rows.extend(self._rows_from_episodes(episodes, advantages))
            self._log_rollout_metrics(episodes, advantages, mode)

        all_rows = self._pad_training_rows(all_rows, mode=mode)
        return self._build_model_inputs(all_rows, mode=mode)

    def _prepare_inputs(
        self, generation_batch: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        """Generate once per group and split turn rows across accumulation steps."""

        mode = "train" if self.model.training else "eval"
        if mode == "eval":
            return self._generate_and_score_completions(generation_batch)

        generate_every = self.args.steps_per_generation * self.num_iterations
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            generated = self._generate_and_score_completions(generation_batch)
            self._buffered_inputs = split_tensor_dict(
                generated, self.args.steps_per_generation
            )
        return self._buffered_inputs[self._step % self.args.steps_per_generation]
