from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
import pytest

import torch

from trl import GRPOTrainer

from rl_training.grpo_trainer import PuzzleGRPOTrainer
from rl_training.sync_rollout import TurnRecord


class FakeAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def gather(value):
        return value


def make_adapter() -> PuzzleGRPOTrainer:
    trainer = object.__new__(PuzzleGRPOTrainer)
    trainer.accelerator = FakeAccelerator()
    trainer._tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.pad_to_multiple_of = None
    trainer.args = SimpleNamespace(
        per_device_train_batch_size=1,
        steps_per_generation=2,
    )
    trainer.model = object()
    trainer.use_vllm = False
    trainer.vllm_importance_sampling_correction = False
    trainer.vllm_importance_sampling_clip_min = 0.95
    trainer.vllm_importance_sampling_clip_max = 1.05
    trainer._metrics = defaultdict(lambda: defaultdict(list))
    trainer._get_per_token_logps_and_entropies = (
        lambda _model, input_ids, _attention_mask, logits_to_keep, **_kwargs: (
            torch.zeros(
                (input_ids.shape[0], logits_to_keep),
                dtype=torch.float32,
            ),
            None,
            None,
        )
    )
    return trainer


def test_turn_rows_have_loss_only_on_generated_tokens() -> None:
    trainer = make_adapter()
    rows = [
        (TurnRecord([1, 2], [3, 4], [-0.1, -0.2]), 0.5, True),
        (TurnRecord([5], [6], [-0.3]), -0.5, False),
    ]

    inputs = trainer._build_model_inputs(rows, mode="train")

    assert inputs["prompt_ids"].shape == (2, 2)
    assert inputs["completion_ids"].shape == (2, 2)
    assert inputs["completion_mask"].tolist() == [[1, 1], [0, 0]]
    assert inputs["generated_token_mask"].tolist() == [[1, 1], [1, 0]]
    assert inputs["advantages"].tolist() == [0.5, -0.5]
    assert inputs["num_items_in_batch"].item() == 2


def test_training_rows_are_padded_for_accumulation_splits() -> None:
    trainer = make_adapter()
    rows = [(TurnRecord([1], [2], [-0.1]), 0.0, True)]

    padded = trainer._pad_training_rows(rows, mode="train")

    assert len(padded) == 2
    assert padded[-1][1:] == (0.0, False)
    assert padded[-1][0].completion_ids == []


def test_train_inference_metrics_include_masked_generated_tokens() -> None:
    trainer = make_adapter()

    trainer._log_train_inference_metrics(
        training_logprobs=torch.tensor([[-0.2, -0.4], [0.0, 0.0]]),
        sampling_logprobs=torch.tensor([[-0.1, -0.5], [0.0, 0.0]]),
        generated_token_mask=torch.tensor([[1, 1], [0, 0]]),
        mode="train",
    )

    metrics = trainer._metrics["train"]
    assert metrics["train_inference/kl_mean"] == [pytest.approx(0.0, abs=1e-7)]
    assert metrics["train_inference/abs_log_ratio_mean"] == [pytest.approx(0.1)]
    assert metrics["train_inference/ratio_outside_clip_fraction"] == [1.0]


def test_adapter_delegates_objective_to_standard_grpo() -> None:
    assert PuzzleGRPOTrainer._compute_loss is GRPOTrainer._compute_loss
