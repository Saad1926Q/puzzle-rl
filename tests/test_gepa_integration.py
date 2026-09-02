from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import gepa
import pytest


from evaluation.constants import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_HISTORY
from evaluation.dataset import PuzzleExample
from evaluation.evaluator import evaluate_episode as canonical_evaluate_episode
from prompt_optimization.adapter import PuzzleGEPAAdapter, STRATEGY_COMPONENT
from prompt_optimization.eval.client import OpenRouterAgent
from prompt_optimization.eval.evaluator import evaluate_episode
from prompt_optimization.eval.protocol import HistoryTurn, build_messages
from prompt_optimization.feedback import episode_reflection_record
from prompt_optimization.runner import OpenRouterRolloutConfig, ensure_reflection_runtime


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


def test_adapter_parallelizes_rollouts_and_preserves_input_order(monkeypatch) -> None:
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_evaluate(item, *_args, **_kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03 if item.example_id == "first" else 0.01)
        with lock:
            active -= 1
        return SimpleNamespace(reward=float(item.optimal_length), example=item)

    monkeypatch.setattr("prompt_optimization.adapter.evaluate_episode", fake_evaluate)
    first = example()
    first = PuzzleExample(
        example_id="first",
        board=first.board,
        optimal_actions=first.optimal_actions,
        optimal_length=1,
        metadata=first.metadata,
    )
    second = PuzzleExample(
        example_id="second",
        board=first.board,
        optimal_actions=first.optimal_actions,
        optimal_length=2,
        metadata=first.metadata,
    )
    adapter = PuzzleGEPAAdapter(
        agent_factory=lambda _prompt: SequenceAgent([]),
        max_turns=45,
        keep_history=False,
        keep_reasoning=False,
        parallelism=2,
    )

    batch = adapter.evaluate(
        [first, second],
        {STRATEGY_COMPONENT: "Use a plan."},
        capture_traces=True,
    )

    assert maximum_active == 2
    assert [output.example.example_id for output in batch.outputs] == ["first", "second"]
    assert batch.scores == [1.0, 2.0]



def test_isolated_openrouter_client_rejects_nonpositive_timeout() -> None:
    with pytest.raises(ValueError, match="request_timeout must be positive"):
        OpenRouterAgent(
            api_key="not-used",
            model="test/model",
            system_prompt="test",
            request_timeout=0,
            client=object(),
        )


def test_isolated_openrouter_client_accepts_runner_provider_options() -> None:
    agent = OpenRouterAgent(
        api_key="not-used",
        model="test/model",
        system_prompt="test",
        upstream_providers=("friendli",),
        allow_fallbacks=False,
        data_collection="deny",
        distillable_only=False,
        quantizations=("fp8",),
        request_timeout=120,
        client=object(),
    )

    assert agent.upstream_providers == ("friendli",)
    assert agent.quantizations == ("fp8",)


def test_invalid_openrouter_response_becomes_episode_api_error() -> None:
    class EmptyCompletions:
        calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(choices=[])

    completions = EmptyCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    agent = OpenRouterAgent(
        api_key="not-used",
        model="test/model",
        system_prompt="test",
        provider_retries=2,
        retry_delay=0,
        client=client,
    )

    episode = evaluate_episode(example(), agent)

    assert completions.calls == 3
    assert episode.outcome == "api_error"
    assert episode.reward == 0
    assert agent.last_response_metadata["error_type"] == (
        "InvalidOpenRouterResponseError"
    )


def test_isolated_openrouter_retries_upstream_provider_error() -> None:
    class ProviderError(RuntimeError):
        status_code = 400
        body = {"error": {"message": "Provider returned error"}}

    class Completions:
        calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("upstream rejected request")
            message = SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="slide_tile", arguments='{"tile": 8}'
                        )
                    )
                ],
                content="",
                reasoning="",
                reasoning_details=None,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                usage=None,
            )

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent = OpenRouterAgent(
        api_key="not-used",
        model="test/model",
        system_prompt="test",
        retry_delay=0,
        client=client,
    )

    assert agent.next_action(example().board) == '{"tile": 8}'
    assert completions.calls == 2


def test_isolated_illegal_attempt_does_not_count_as_move() -> None:
    isolated = evaluate_episode(example(), SequenceAgent(['{"tile": 1}']))
    canonical = canonical_evaluate_episode(example(), SequenceAgent(['{"tile": 1}']))

    assert isolated.outcome == canonical.outcome == "illegal"
    assert isolated.moves_taken == canonical.moves_taken == 0


def test_reflection_record_handles_already_solved_board() -> None:
    solved_example = PuzzleExample(
        example_id="solved",
        board=(1, 2, 3, 4, 5, 6, 7, 8, 0),
        optimal_actions=(),
        optimal_length=0,
        metadata={"bucket": "test", "action_interface": "tile_id_v1"},
    )
    episode = evaluate_episode(solved_example, SequenceAgent([]))

    record = episode_reflection_record(episode)

    assert record["Generated Outputs"]["reasoning_excerpts"] == []
    assert record["Generated Outputs"]["final_distance"] == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"request_timeout": 0}, "request_timeout"),
        ({"max_turns": 46}, "max_turns"),
        ({"top_p": 0}, "top_p"),
        ({"keep_history": False, "keep_reasoning": True}, "keep_reasoning"),
    ],
)
def test_rollout_config_rejects_invalid_settings(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        OpenRouterRolloutConfig(model="test/model", **overrides)


def test_missing_litellm_fails_before_gepa_evaluation(monkeypatch) -> None:
    def missing_litellm(_name: str) -> None:
        error = ModuleNotFoundError("No module named 'litellm'")
        error.name = "litellm"
        raise error

    monkeypatch.setattr(
        "prompt_optimization.runner.importlib.import_module", missing_litellm
    )

    with pytest.raises(RuntimeError, match="uv sync"):
        ensure_reflection_runtime()

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
