from __future__ import annotations

from types import SimpleNamespace

import pytest

import rl_training.sync_rollout as sync_rollout
from puzzle3.environment import PuzzleEnv


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self) -> None:
        self.messages: list[list[dict[str, object]]] = []

    def apply_chat_template(self, messages, **_kwargs):
        self.messages.append(messages)
        return list(range(10 + len(messages)))


class FakeVLLM:
    def __init__(self) -> None:
        self.calls = 0

    def sync_weights(self) -> None:
        pass

    def generate(self, *, prompts, images, num_generations):
        assert images is None
        assert num_generations == 1
        tile = 7 if self.calls == 0 else 8
        self.calls += 1
        completion_ids = [[tile] for _ in prompts]
        logprobs = [[[-0.25]] for _ in prompts]
        return prompts, completion_ids, logprobs, None


@pytest.fixture
def fake_trainer(monkeypatch) -> SimpleNamespace:
    tokenizer = FakeTokenizer()

    def fake_parse_response(_tokenizer, completion_ids, *, prefix):
        assert prefix
        return {
            "role": "assistant",
            "content": "move",
            "tool_calls": [
                {
                    "function": {
                        "name": "slide_tile",
                        "arguments": {"tile": completion_ids[0]},
                    }
                }
            ],
        }

    monkeypatch.setattr(sync_rollout, "parse_response", fake_parse_response)
    probe = PuzzleEnv()
    return SimpleNamespace(
        processing_class=tokenizer,
        chat_template=None,
        chat_template_kwargs={},
        _env_tools={None: [probe.slide_tile]},
        history_turns=1,
        use_vllm=True,
        state=SimpleNamespace(global_step=0),
        _last_loaded_step=-1,
        vllm_generation=FakeVLLM(),
    )


def test_sync_rollout_rebuilds_bounded_history(fake_trainer) -> None:
    episodes = sync_rollout.generate_episode_group(
        fake_trainer,
        [
            {"board": [1, 2, 3, 4, 5, 6, 0, 7, 8], "optimal_length": 2, "max_turns": 2},
            {"board": [1, 2, 3, 4, 5, 6, 0, 7, 8], "optimal_length": 2, "max_turns": 2},
        ],
    )

    assert [episode.outcome for episode in episodes] == ["solved", "solved"]
    assert all(episode.reward == pytest.approx(1.0) for episode in episodes)
    assert len(episodes[0].turns) == 2
    assert len(fake_trainer.processing_class.messages) == 4
    assert len(fake_trainer.processing_class.messages[0]) == 2
    assert len(fake_trainer.processing_class.messages[2]) == 4
    assert episodes[0].turns[0].logprobs == [-0.25]
    assert episodes[0].turns[1].prompt_ids != episodes[0].turns[0].prompt_ids


def test_tool_tile_rejects_wrong_tool_and_arguments() -> None:
    assert sync_rollout._tool_tile({"tool_calls": []}) is None
    assert (
        sync_rollout._tool_tile(
            {
                "tool_calls": [
                    {"function": {"name": "other", "arguments": {"tile": 1}}}
                ]
            }
        )
        is None
    )
    assert (
        sync_rollout._tool_tile(
            {
                "tool_calls": [
                    {"function": {"name": "slide_tile", "arguments": {"tile": True}}}
                ]
            }
        )
        is None
    )
