from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from evaluation.dataset import PuzzleExample
from evaluation.evaluator import evaluate, evaluate_episode
from evaluation.results import EpisodeResult, EvaluationResult, StepResult
from evaluation.rewards import distance_progress_reward, solved_reward
from evaluation.protocol import HistoryTurn
from puzzle3.board import GOAL
from puzzle3.solver import exact_distance


@dataclass
class SequenceAgent:
    responses: list[str]

    def next_action(self, board: tuple[int, ...]) -> str:
        return self.responses.pop(0)

def example(board: tuple[int, ...], optimal_length: int = 1) -> PuzzleExample:
    return PuzzleExample("test", board, tuple(), optimal_length, {})

def test_evaluator_keeps_legacy_result_and_reward_exports() -> None:
    from evaluation import evaluator

    assert evaluator.StepResult is StepResult
    assert evaluator.EpisodeResult is EpisodeResult
    assert evaluator.EvaluationResult is EvaluationResult
    assert evaluator.solved_reward is solved_reward
    assert evaluator.distance_progress_reward is distance_progress_reward

def test_history_is_bounded_to_four_completed_turns() -> None:
    @dataclass
    class HistoryAgent:
        responses: list[str]
        received_history: list[tuple[HistoryTurn, ...]]

        def next_action(
            self,
            board: tuple[int, ...],
            history: tuple[HistoryTurn, ...],
            *,
            include_reasoning: bool,
        ) -> str:
            assert include_reasoning is False
            self.received_history.append(history)
            return self.responses.pop(0)
    agent = HistoryAgent(
        [
            '{"tile": 7}',
            '{"tile": 4}',
            '{"tile": 1}',
            '{"tile": 2}',
            '{"tile": 3}',
        ],
        [],
    )
    evaluate_episode(
        example((1, 2, 3, 4, 5, 6, 7, 0, 8)),
        agent,
        max_turns=5,
        keep_history=True,
    )

    assert [len(history) for history in agent.received_history] == [0, 1, 2, 3, 4]
    assert [turn.tile for turn in agent.received_history[-1]] == [7, 4, 1, 2]

def test_history_retains_reasoning_only_when_requested() -> None:
    @dataclass
    class ReasoningAgent:
        calls: int = 0
        received_history: list[tuple[HistoryTurn, ...]] = field(default_factory=list)

        def __post_init__(self) -> None:
            self.last_response_metadata: dict[str, object] = {}

        def next_action(
            self,
            board: tuple[int, ...],
            history: tuple[HistoryTurn, ...],
            *,
            include_reasoning: bool,
        ) -> str:
            self.received_history.append(history)
            self.last_response_metadata = {
                "reasoning_content": "Move toward goal.",
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "Move toward goal."}
                ],
            }
            self.calls += 1
            return '{"tile": 7}' if self.calls == 1 else '{"tile": 4}'

    agent = ReasoningAgent()
    evaluate_episode(
        example((1, 2, 3, 4, 5, 6, 7, 0, 8)),
        agent,
        max_turns=2,
        keep_history=True,
        keep_reasoning=True,
    )

    assert agent.received_history[1][0].reasoning == "Move toward goal."
    assert agent.received_history[1][0].reasoning_details == [
        {"type": "reasoning.text", "text": "Move toward goal."}
    ]

def test_reasoning_history_requires_history() -> None:
    with pytest.raises(ValueError, match="keep_reasoning requires keep_history"):
        evaluate_episode(
            example((1, 2, 3, 4, 5, 6, 7, 0, 8)),
            SequenceAgent(['{"tile": 8}']),
            keep_reasoning=True,
        )

def test_evaluation_records_api_error_without_aborting_other_rollouts() -> None:
    class APIErrorAgent:
        last_response_metadata: dict[str, object] = {}

        def next_action(self, board):
            self.last_response_metadata = {
                "status": "api_error",
                "error_type": "BadRequestError",
                "status_code": 400,
            }
            raise RuntimeError("provider unavailable")

    result = evaluate(
        [example((1, 2, 3, 4, 5, 6, 7, 0, 8))],
        APIErrorAgent(),
        num_rollouts=2,
    )

    assert len(result.episodes) == 2
    assert result.summary()["api_error"] == 2
    assert result.summary()["api_error_rate"] == 1.0
    assert result.summary()["mean_reward"] == 0.0

def test_solved_reward_is_bounded_and_efficiency_sensitive() -> None:
    assert solved_reward(4, 4) == 1.0
    assert solved_reward(4, 8) == 0.9
    assert solved_reward(4, 100) == 0.808

def test_exact_distance_is_goal_oriented() -> None:
    one_move = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    hardest = (8, 6, 7, 2, 5, 4, 3, 0, 1)
    assert exact_distance(GOAL) == 0
    assert exact_distance(one_move) == 1
    assert exact_distance(hardest) == 31

def test_distance_progress_rewards_progress_and_penalizes_backtracking() -> None:
    one_move = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    farther = (1, 2, 3, 4, 5, 6, 0, 7, 8)
    closer = distance_progress_reward(one_move, GOAL)
    farther_reward = distance_progress_reward(one_move, farther)
    assert closer == pytest.approx(0.25 / 31)
    assert farther_reward == pytest.approx(-0.25 / 31)
    assert closer + farther_reward == pytest.approx(0.0)

def test_solved_episode_includes_terminal_and_progress_rewards() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, SequenceAgent(['{"tile": 8}']))
    assert result.outcome == "solved"
    assert result.reward == pytest.approx(1.0 + 0.25 / 31)
    assert result.steps[0].terminal_reward == pytest.approx(1.0)
    assert result.steps[0].progress_reward == pytest.approx(0.25 / 31)
    assert result.steps[0].legal_tiles == (5, 7, 8)
    assert result.steps[0].tile == 8
    assert result.final_board == (1, 2, 3, 4, 5, 6, 7, 8, 0)

def test_serialized_steps_contain_no_directional_move_field() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, SequenceAgent(['{"tile": 8}']))
    assert "move" not in result.steps[0].to_dict()

def test_illegal_move_ends_immediately_with_negative_reward() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, SequenceAgent(['{"tile": 1}']))
    assert result.outcome == "illegal"
    assert result.reward == -1.0
    assert result.steps[0].progress_reward == 0.0
    assert result.steps[0].terminal_reward == -1.0
    assert result.steps[0].legal_tiles == (5, 7, 8)
    assert result.steps[0].tile == 1
    assert len(result.steps) == 1

def test_malformed_response_ends_immediately_with_negative_reward() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, SequenceAgent(["I choose left"]))
    assert result.outcome == "malformed"
    assert result.reward == -1.0
    assert result.steps[0].progress_reward == 0.0
    assert result.moves_taken == 0

def test_truncated_response_is_reported_separately() -> None:
    class TruncatedAgent:
        last_response_metadata = {"truncated": True}

        def next_action(self, board: tuple[int, ...]) -> str:
            return ""

    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, TruncatedAgent())
    assert result.outcome == "truncated"
    assert result.steps[0].status == "truncated"
    assert result.reward == -1.0
    assert result.steps[0].progress_reward == 0.0
    assert evaluate([task], TruncatedAgent()).summary()["truncated"] == 1

def test_multiple_rollouts_report_rollout_metrics_and_pass_at_k() -> None:
    class FirstFailsThenSolves:
        def __init__(self) -> None:
            self.calls = 0

        def next_action(self, board: tuple[int, ...]) -> str:
            self.calls += 1
            return "bad" if self.calls == 1 else '{"tile": 8}'

    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate([task], FirstFailsThenSolves(), num_rollouts=3)
    summary = result.summary()
    assert len(result.episodes) == 3
    assert summary["num_examples"] == 1
    assert summary["num_rollouts"] == 3
    assert summary["num_episodes"] == 3
    assert summary["solved"] == 2
    assert summary["malformed"] == 1
    assert summary["pass@k"] == 1.0
    assert [episode.rollout_id for episode in result.episodes] == [0, 1, 2]

def test_parallel_rollouts_preserve_order_and_isolate_agents() -> None:
    import threading

    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    lock = threading.Lock()
    created = 0

    class WorkerAgent:
        def __init__(self, worker_id: int) -> None:
            self.worker_id = worker_id
            self.last_response_metadata = {"worker_id": worker_id}

        def next_action(self, board: tuple[int, ...]) -> str:
            return '{"tile": 8}'

    def factory() -> WorkerAgent:
        nonlocal created
        with lock:
            worker_id = created
            created += 1
        return WorkerAgent(worker_id)

    result = evaluate(
        [task, task], num_rollouts=2, parallelism=2, agent_factory=factory
    )
    assert len(result.episodes) == 4
    assert [
        (episode.example.example_id, episode.rollout_id) for episode in result.episodes
    ] == [("test", 0), ("test", 1), ("test", 0), ("test", 1)]
    assert {
        episode.steps[0].response_metadata["worker_id"] for episode in result.episodes
    } == {0, 1, 2, 3}

def test_parallelism_requires_agent_factory() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    with pytest.raises(ValueError, match="agent_factory is required"):
        evaluate([task], SequenceAgent(['{"tile": 8}']), parallelism=2)

def test_num_rollouts_must_be_positive() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    with pytest.raises(ValueError, match="num_rollouts must be positive"):
        evaluate([task], SequenceAgent(['{"tile": 8}']), num_rollouts=0)

def test_valid_unsolved_trajectory_at_limit_gets_progress_and_timeout_penalty() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, SequenceAgent(['{"tile": 7}']), max_turns=1)
    assert result.outcome == "timeout"
    assert result.reward == pytest.approx(-0.25 - 0.25 / 31)
    assert result.steps[0].progress_reward == pytest.approx(-0.25 / 31)
    assert result.steps[0].terminal_reward == pytest.approx(-0.25)
    assert result.moves_taken == 1

def test_turn_limit_cannot_exceed_45() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    with pytest.raises(ValueError, match="between 1 and 45"):
        evaluate_episode(task, SequenceAgent([]), max_turns=46)

