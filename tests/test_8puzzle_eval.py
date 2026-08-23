from __future__ import annotations

from dataclasses import dataclass
import json

import pytest
from datasets import Dataset

from evaluation.dataset import DatasetError, PuzzleExample, load_examples
from evaluation.evaluator import (
    evaluate,
    evaluate_episode,
    distance_progress_reward,
    solved_reward,
)
from evaluation.clients.deepseek import DeepSeekAgent
from evaluation.clients.glm import GLMAgent
from evaluation.clients.openai_client import OpenAIAgent
from evaluation.clients.qwen import QwenAgent
from evaluation.protocol import build_messages, get_api_key, parse_tile
from puzzle3.board import GOAL
from puzzle3.solver import exact_distance


@dataclass
class SequenceAgent:
    responses: list[str]

    def next_action(self, board: tuple[int, ...]) -> str:
        return self.responses.pop(0)


def example(board: tuple[int, ...], optimal_length: int = 1) -> PuzzleExample:
    return PuzzleExample("test", board, tuple(), optimal_length, {})


def test_parser_requires_one_valid_tile() -> None:
    assert parse_tile('{"tile": 8}') == 8
    assert parse_tile('{"tile": 0}') is None
    assert parse_tile('{"tile": 9}') is None
    assert parse_tile('{"tile": true}') is None
    assert parse_tile('{"move": "left"}') is None


def test_messages_contain_current_board_but_no_history() -> None:
    messages = build_messages((1, 2, 3, 4, 5, 6, 7, 0, 8))
    assert len(messages) == 2
    assert "1 2 3" in messages[1]["content"]
    assert "history" not in messages[1]["content"].lower()


def _fake_response(
    *, content="", tool_calls=None, finish_reason="tool_calls", reasoning_content=None
):
    message = type(
        "Message",
        (),
        {
            "content": content,
            "tool_calls": tool_calls,
            "reasoning_content": reasoning_content,
        },
    )()
    choice = type("Choice", (), {"message": message, "finish_reason": finish_reason})()
    return type("Response", (), {"choices": [choice], "usage": None})()


def _fake_tool_call(arguments='{"tile":8}', name="slide_tile"):
    function = type("Function", (), {"name": name, "arguments": arguments})()
    return [type("ToolCall", (), {"function": function})()]


def _fake_responses_response(*, arguments='{"tile":8}', status="completed"):
    call = type(
        "FunctionCall",
        (),
        {"type": "function_call", "name": "slide_tile", "arguments": arguments},
    )()
    incomplete = type("Incomplete", (), {"reason": "max_output_tokens"})()
    return type(
        "Response",
        (),
        {
            "output": [call],
            "output_text": "",
            "status": status,
            "incomplete_details": incomplete if status == "incomplete" else None,
            "usage": None,
        },
    )()


def test_deepseek_strict_tool_call_is_canonicalized() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(tool_calls=_fake_tool_call())

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert parse_tile(agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))) == 8
    assert completions.kwargs["tool_choice"]["function"]["name"] == "slide_tile"
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert agent.last_response_metadata["status"] == "tool_call"


def test_glm_tool_call_uses_zai_parameters() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(tool_calls=_fake_tool_call('{"tile":3}'))

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = GLMAgent(api_key="not-used", client=client, thinking=True)
    assert parse_tile(agent.next_action((1, 2, 0, 4, 5, 3, 7, 8, 6))) == 3
    assert completions.kwargs["tool_choice"] == "auto"
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "reasoning_effort" not in completions.kwargs
    assert completions.kwargs["max_tokens"] == 4096


def test_glm_disabled_thinking_is_explicit() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(tool_calls=_fake_tool_call())

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = GLMAgent(api_key="not-used", client=client, thinking=False)
    agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_qwen_uses_native_tool_call_and_sampling_parameters() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(tool_calls=_fake_tool_call('{"tile":8}'))

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = QwenAgent(
        client=client,
        thinking=True,
        max_tokens=256,
        temperature=0.8,
        top_p=0.9,
        top_k=10,
        presence_penalty=1.5,
        repetition_penalty=1.1,
    )

    assert parse_tile(agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))) == 8
    assert completions.kwargs["tools"][0]["function"]["name"] == "slide_tile"
    assert completions.kwargs["tool_choice"] == "auto"
    assert completions.kwargs["parallel_tool_calls"] is False
    assert completions.kwargs["max_tokens"] == 256
    assert completions.kwargs["temperature"] == 0.8
    assert completions.kwargs["top_p"] == 0.9
    assert completions.kwargs["presence_penalty"] == 1.5
    assert completions.kwargs["extra_body"] == {
        "top_k": 10,
        "repetition_penalty": 1.1,
        "enable_thinking": True,
    }
    assert agent.last_response_metadata["status"] == "tool_call"


def test_qwen_defaults_to_non_thinking_mode() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(tool_calls=_fake_tool_call())

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()

    QwenAgent(client=client).next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))

    assert completions.kwargs["extra_body"]["enable_thinking"] is False


def test_qwen_cli_uses_local_defaults_and_needs_no_api_key() -> None:
    import runpy

    runner = runpy.run_path("scripts/run_eval_8puzzle.py")
    args = runner["build_parser"]().parse_args(["--provider", "qwen"])
    runner["resolve_provider_args"](args)

    assert args.model == "Qwen/Qwen3.5-0.8B"
    assert args.base_url == "http://localhost:8000/v1"
    assert args.api_key_env is None
    assert args.thinking is False


def test_openai_responses_tool_call_uses_responses_parameters() -> None:
    class Responses:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_responses_response(arguments='{"tile":6}')

    responses = Responses()
    client = type("Client", (), {"responses": responses})()
    agent = OpenAIAgent(api_key="not-used", client=client, thinking=True)
    assert parse_tile(agent.next_action((1, 2, 3, 4, 5, 0, 7, 8, 6))) == 6
    assert responses.kwargs["tool_choice"]["name"] == "slide_tile"
    assert responses.kwargs["parallel_tool_calls"] is False
    assert responses.kwargs["max_output_tokens"] == 4096
    assert responses.kwargs["reasoning"]["effort"] == "low"
    assert responses.kwargs["store"] is False


def test_openai_responses_truncated_tool_call_is_not_accepted() -> None:
    class Responses:
        def create(self, **kwargs):
            return _fake_responses_response(status="incomplete")

    client = type("Client", (), {"responses": Responses()})()
    agent = OpenAIAgent(api_key="not-used", client=client)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["truncated"] is True
    assert agent.last_response_metadata["status"] == "truncated"
    assert agent.last_response_metadata["incomplete_reason"] == "max_output_tokens"


def test_deepseek_thinking_mode_reads_tool_call_and_reasoning_metadata() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(
                tool_calls=_fake_tool_call('{"tile":3}'),
                reasoning_content="reason about the board",
            )

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=True)
    assert parse_tile(agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))) == 3
    assert completions.kwargs["tool_choice"] == "auto"
    assert agent.last_response_metadata["reasoning_content_length"] > 0


def test_deepseek_multiple_tool_calls_are_malformed() -> None:
    class Completions:
        def create(self, **kwargs):
            return _fake_response(
                tool_calls=_fake_tool_call('{"tile":8}') + _fake_tool_call('{"tile":3}')
            )

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["tool_call_count"] == 2
    assert agent.last_response_metadata["status"] == "malformed"


def test_deepseek_truncated_tool_call_is_not_accepted() -> None:
    class Completions:
        def create(self, **kwargs):
            return _fake_response(
                tool_calls=_fake_tool_call('{"tile":8}'), finish_reason="length"
            )

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["truncated"] is True
    assert agent.last_response_metadata["status"] == "truncated"


def test_deepseek_empty_or_truncated_response_is_diagnosed() -> None:
    class Completions:
        def create(self, **kwargs):
            return _fake_response(
                content="",
                tool_calls=None,
                finish_reason="length",
                reasoning_content="partial",
            )

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=True)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["status"] == "truncated"
    assert agent.last_response_metadata["truncated"] is True
    assert agent.last_response_metadata["reasoning_content_length"] == 7


def test_deepseek_malformed_tool_arguments_are_diagnosed() -> None:
    class Completions:
        def create(self, **kwargs):
            return _fake_response(tool_calls=_fake_tool_call('{"tile":0}'))

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["tool_name"] == "slide_tile"
    assert agent.last_response_metadata["status"] == "malformed"


def test_deepseek_provider_error_is_not_relabelled_as_malformed() -> None:
    class Completions:
        def create(self, **kwargs):
            raise RuntimeError("provider unavailable")

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))
    assert agent.last_response_metadata["status"] == "api_error"


def test_deepseek_adapter_starts_a_fresh_request_each_turn() -> None:
    class Completions:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            response = type("Response", (), {})()
            response.choices = [
                type(
                    "Choice",
                    (),
                    {"message": type("Message", (), {"content": '{"tile": 8}'})()},
                )()
            ]
            return response

    completions = Completions()
    client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    agent = DeepSeekAgent(api_key="not-used", client=client)
    agent.next_action((1, 2, 3, 4, 5, 6, 7, 0, 8))
    agent.next_action((1, 2, 3, 4, 5, 6, 0, 7, 8))
    assert len(completions.calls) == 2
    assert len(completions.calls[0]["messages"]) == 2
    assert completions.calls[0]["messages"][1] != completions.calls[1]["messages"][1]
    assert all(len(call["messages"]) == 2 for call in completions.calls)


def test_local_jsonl_eval_subset_is_supported(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "board": [1, 2, 3, 4, 5, 6, 7, 0, 8],
                "action_interface": "tile_id_v1",
                "optimal_actions": [8],
                "optimal_length": 1,
                "bucket": "easy",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tasks = load_examples(dataset=str(path))
    assert len(tasks) == 1
    assert tasks[0].board == (1, 2, 3, 4, 5, 6, 7, 0, 8)
    assert tasks[0].metadata["bucket"] == "easy"


def test_legacy_directional_dataset_rows_are_rejected(tmp_path) -> None:
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        json.dumps(
            {
                "board": [1, 2, 3, 4, 5, 6, 7, 0, 8],
                "optimal_moves": ["left"],
                "optimal_length": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="legacy optimal_moves is unsupported"):
        load_examples(dataset=str(path))


def test_local_parquet_eval_subset_is_supported(tmp_path) -> None:
    path = tmp_path / "eval.parquet"
    Dataset.from_list(
        [
            {
                "board": [1, 2, 3, 4, 5, 6, 7, 0, 8],
                "action_interface": "tile_id_v1",
                "optimal_actions": [8],
                "optimal_length": 1,
                "bucket": "easy",
            }
        ]
    ).to_parquet(str(path))

    tasks = load_examples(dataset=str(path))

    assert len(tasks) == 1
    assert tasks[0].board == (1, 2, 3, 4, 5, 6, 7, 0, 8)
    assert tasks[0].metadata["bucket"] == "easy"


def test_dotenv_key_is_supported(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("DEEPSEEK_API_KEY='from-dotenv'\n", encoding="utf-8")
    assert get_api_key(dotenv_path=dotenv) == "from-dotenv"


def test_environment_key_takes_precedence_over_dotenv(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("DEEPSEEK_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-environment")
    assert get_api_key(dotenv_path=dotenv) == "from-environment"


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


def test_harness_runs_ten_huggingface_dataset_examples(monkeypatch) -> None:
    from evaluation import dataset as dataset_module

    rows = [
        {
            "board": [1, 2, 3, 4, 5, 6, 7, 0, 8],
            "action_interface": "tile_id_v1",
            "optimal_actions": [8],
            "optimal_length": 1,
        }
    ] * 10
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: rows)
    tasks = load_examples(limit=10)

    # Use a fresh oracle that derives the next action from the current board by BFS;
    # this makes the smoke test independent of any model or credentials.
    from puzzle3.solver import solve

    class BfsOracle:
        def next_action(self, board: tuple[int, ...]) -> str:
            return json.dumps({"tile": solve(board)[0]})

    result = evaluate(tasks, BfsOracle())
    assert len(result.episodes) == 10
    assert result.summary()["solved"] == 10
    assert result.summary()["mean_reward"] == pytest.approx(1.0 + 0.25 / 31)
