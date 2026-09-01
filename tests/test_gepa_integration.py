from __future__ import annotations

import json

import gepa

from evaluation.constants import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_HISTORY
from evaluation.dataset import PuzzleExample
from evaluation.evaluator import evaluate_episode as canonical_evaluate_episode
from prompt_optimization.adapter import PuzzleGEPAAdapter, STRATEGY_COMPONENT
from prompt_optimization.eval.evaluator import evaluate_episode
from prompt_optimization.eval.protocol import HistoryTurn, build_messages
from prompt_optimization.feedback import episode_reflection_record


class SequenceAgent:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.last_response_metadata: dict[str, object] = {
            "usage": {"completion_tokens": 7},
            "reasoning_content": "Follow the current board.",
        }

    def next_action(self, *_args, **_kwargs) -> str:
        return next(self.responses)


def example() -> PuzzleExample:
    return PuzzleExample(
        example_id="gepa-test",
        board=(1, 2, 3, 4, 5, 6, 7, 0, 8),
        optimal_actions=(8,),
        optimal_length=1,
        metadata={"bucket": "bucket_1", "action_interface": "tile_id_v1"},
    )


def test_adapter_injects_only_candidate_strategy_and_uses_native_reward() -> None:
    prompts: list[str] = []

    def factory(prompt: str) -> SequenceAgent:
        prompts.append(prompt)
        return SequenceAgent(['{"tile": 8}'])

    adapter = PuzzleGEPAAdapter(
        agent_factory=factory,
        max_turns=45,
        keep_history=True,
        keep_reasoning=True,
    )
    batch = adapter.evaluate(
        [example()], {STRATEGY_COMPONENT: "Use a short verified plan."}, capture_traces=True
    )

    direct = canonical_evaluate_episode(example(), SequenceAgent(['{"tile": 8}']))
    assert batch.scores == [direct.reward]
    assert batch.trajectories == batch.outputs
    assert prompts == [
        "You solve one 3x3 sliding puzzle one move at a time.\n"
        "The solved board is 1 2 3 / 4 5 6 / 7 8 0. On each turn, slide one numbered\n"
        "tile adjacent to the blank (0) into the blank by calling slide_tile exactly\n"
        "once with that tile's number. Only a tile directly adjacent to the blank is a\n"
        "legal move; the blank itself, a non-adjacent tile, or more than one tile per\n"
        "turn is illegal and will be rejected.\nUse a short verified plan."
    ]


def test_reflection_record_excludes_optimal_actions_and_summarizes_episode() -> None:
    episode = evaluate_episode(example(), SequenceAgent(['{"tile": 8}']))
    record = episode_reflection_record(episode)
    serialized = json.dumps(record)

    assert "optimal_actions" not in serialized
    assert record["Inputs"]["optimal_depth"] == 1
    assert record["Generated Outputs"]["outcome"] == "solved"
    assert record["Generated Outputs"]["action_trace"][0]["distance_before"] == 1
    assert "Outcome: solved" in record["Feedback"]


def test_gepa_smoke_mutates_external_strategy_with_fake_agents(tmp_path) -> None:
    def factory(prompt: str) -> SequenceAgent:
        if "improved" in prompt:
            return SequenceAgent(['{"tile": 8}'])
        return SequenceAgent(['{"tile": 1}'])

    adapter = PuzzleGEPAAdapter(
        agent_factory=factory,
        max_turns=45,
        keep_history=False,
        keep_reasoning=False,
    )

    def proposer(candidate, reflective_dataset, components_to_update, **_kwargs):
        assert components_to_update == [STRATEGY_COMPONENT]
        assert reflective_dataset[STRATEGY_COMPONENT]
        return {STRATEGY_COMPONENT: "Use an improved short plan."}

    result = gepa.optimize(
        seed_candidate={STRATEGY_COMPONENT: "Use a plan."},
        trainset=[example()],
        valset=[example()],
        adapter=adapter,
        custom_candidate_proposer=proposer,
        max_metric_calls=4,
        reflection_minibatch_size=1,
        run_dir=str(tmp_path / "gepa"),
        seed=42,
        use_merge=False,
        track_best_outputs=False,
    )

    assert result.num_candidates >= 2
    assert result.best_candidate[STRATEGY_COMPONENT] == "Use an improved short plan."


def test_isolated_rollout_keeps_canonical_history_prompt_unchanged() -> None:
    history = (HistoryTurn(example().board, tile=8),)
    messages = build_messages(
        example().board,
        "Isolated candidate strategy.",
        history,
    )

    assert SYSTEM_PROMPT_WITH_HISTORY == SYSTEM_PROMPT + (
        "\nThe latest board observation is authoritative; retained history is "
        "supplementary context."
    )
    assert messages[0]["content"] == "Isolated candidate strategy."
    assert messages[1]["role"] == "user"


def test_isolated_reward_semantics_match_canonical_evaluator() -> None:
    isolated = evaluate_episode(example(), SequenceAgent(['{"tile": 8}']))
    canonical = canonical_evaluate_episode(example(), SequenceAgent(['{"tile": 8}']))

    assert isolated.reward == canonical.reward
    assert isolated.outcome == canonical.outcome
    assert isolated.moves_taken == canonical.moves_taken
