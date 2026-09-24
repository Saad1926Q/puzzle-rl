from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import Any

from puzzle3.environment import PuzzleEnv
from rl_training.async_worker import _PuzzleAsyncRolloutLoop


class FakeTokenizer:
    eos_token = ""
    eos_token_id = 99
    pad_token_id = 0
    response_template = None

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.prompts: list[list[dict[str, Any]]] = []

    def apply_chat_template(
        self, messages: list[dict[str, Any]], **_: Any
    ) -> list[int]:
        self.prompts.append(messages)
        return [len(self.prompts)]

    def parse_response(self, _ids: list[int]) -> dict[str, Any]:
        return self.responses.pop(0)


def test_async_loop_rebuilds_prompt_from_last_four_turns() -> None:
    tokenizer = FakeTokenizer(
        [
            {
                "role": "assistant",
                "content": "first",
                "tool_calls": [
                    {"function": {"name": "slide_tile", "arguments": {"tile": 7}}}
                ],
            },
            {
                "role": "assistant",
                "content": "second",
                "tool_calls": [
                    {"function": {"name": "slide_tile", "arguments": {"tile": 8}}}
                ],
            },
        ]
    )
    env = PuzzleEnv()
    env.reset((1, 2, 3, 4, 5, 6, 0, 7, 8), optimal_length=2)
    loop = object.__new__(_PuzzleAsyncRolloutLoop)
    loop.tokenizer = tokenizer
    loop.chat_template = None
    loop.chat_template_kwargs = {}
    loop.max_tool_calling_iterations = 5
    loop.history_turns = 4
    loop._fork_threshold_tokens = 1024
    loop._counters = defaultdict(float)
    loop._tool_pool = ThreadPoolExecutor(max_workers=1)
    loop._rates = defaultdict(lambda: [0.0, 0.0])
    loop._push_rollout_metrics = lambda **_: None

    async def generate_one_turn(
        _prompt_ids: list[int],
    ) -> tuple[list[int], list[float]]:
        return [2], [0.0]

    loop._generate_one_turn = generate_one_turn
    completion, completion_ids, sequences, calls, failures, reward = asyncio.run(
        loop._generate_one([], {"slide_tile": env.slide_tile}, [env.slide_tile])
    )
    loop._tool_pool.shutdown(wait=True)

    assert env.outcome == "solved"
    assert calls == 2
    assert failures == 0
    assert reward is None
    assert completion_ids == [2, 2]
    assert len(sequences) == 2
    assert len(completion) == 4
    assert len(tokenizer.prompts) == 2
    second_prompt = tokenizer.prompts[1]
    assert second_prompt[0]["role"] == "system"
    assert second_prompt[0]["content"].startswith("You solve one 3x3 sliding puzzle")
    assert second_prompt[2]["tool_calls"][0]["function"]["arguments"] == '{"tile": 7}'
    assert "Board after that action" in second_prompt[3]["content"]
