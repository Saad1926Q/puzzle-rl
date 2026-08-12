from __future__ import annotations

from dataclasses import dataclass

import pytest

from evaluation.dataset import PuzzleExample, load_examples
from evaluation.evaluator import evaluate, evaluate_episode, solved_reward
from evaluation.model import DeepSeekAgent, build_messages, get_api_key, parse_move


@dataclass
class SequenceAgent:
    responses: list[str]

    def next_move(self, board: tuple[int, ...]) -> str:
        return self.responses.pop(0)


def example(board: tuple[int, ...], optimal_length: int = 1) -> PuzzleExample:
    return PuzzleExample("test", board, tuple(), optimal_length, {})


def test_parser_requires_one_move_tag() -> None:
    assert parse_move("<move> LEFT </move>") == "left"
    assert parse_move("reasoning\n<move>up</move>") == "up"
    assert parse_move("left") is None
    assert parse_move("<move>up</move><move>left</move>") is None
    assert parse_move("<move>jump</move>") is None


def test_messages_contain_current_board_but_no_history() -> None:
    messages = build_messages((1, 2, 3, 4, 5, 6, 7, 0, 8))
    assert len(messages) == 2
    assert "1 2 3" in messages[1]["content"]
    assert "history" not in messages[1]["content"].lower()


def _fake_response(*, content="", tool_calls=None, finish_reason="tool_calls", reasoning_content=None):
    message = type(
        "Message",
        (),
        {"content": content, "tool_calls": tool_calls, "reasoning_content": reasoning_content},
    )()
    choice = type("Choice", (), {"message": message, "finish_reason": finish_reason})()
    return type("Response", (), {"choices": [choice], "usage": None})()


def _fake_tool_call(arguments='{"move":"left"}', name="submit_move"):
    function = type("Function", (), {"name": name, "arguments": arguments})()
    return [type("ToolCall", (), {"function": function})()]


def test_deepseek_strict_tool_call_is_canonicalized() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(tool_calls=_fake_tool_call())

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert parse_move(agent.next_move((1, 2, 3, 4, 5, 6, 7, 0, 8))) == "left"
    assert completions.kwargs["tool_choice"]["function"]["name"] == "submit_move"
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert agent.last_response_metadata["status"] == "tool_call"


def test_deepseek_thinking_mode_reads_tool_call_and_reasoning_metadata() -> None:
    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return _fake_response(
                tool_calls=_fake_tool_call('{"move":"up"}'),
                reasoning_content="reason about the board",
            )

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=True)
    assert parse_move(agent.next_move((1, 2, 3, 4, 5, 6, 7, 0, 8))) == "up"
    assert completions.kwargs["tool_choice"] == "auto"
    assert agent.last_response_metadata["reasoning_content_length"] > 0


def test_deepseek_multiple_tool_calls_are_malformed() -> None:
    class Completions:
        def create(self, **kwargs):
            return _fake_response(
                tool_calls=_fake_tool_call('{"move":"left"}') + _fake_tool_call('{"move":"up"}')
            )

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert agent.next_move((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["tool_call_count"] == 2
    assert agent.last_response_metadata["status"] == "malformed"


def test_deepseek_truncated_tool_call_is_not_accepted() -> None:
    class Completions:
        def create(self, **kwargs):
            return _fake_response(
                tool_calls=_fake_tool_call('{"move":"left"}'), finish_reason="length"
            )

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert agent.next_move((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["truncated"] is True
    assert agent.last_response_metadata["status"] == "truncated"


def test_deepseek_empty_or_truncated_response_is_diagnosed() -> None:
    class Completions:
        def create(self, **kwargs):
            return _fake_response(content="", tool_calls=None, finish_reason="length", reasoning_content="partial")

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=True)
    assert agent.next_move((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["status"] == "truncated"
    assert agent.last_response_metadata["truncated"] is True
    assert agent.last_response_metadata["reasoning_content_length"] == 7


def test_deepseek_malformed_tool_arguments_are_diagnosed() -> None:
    class Completions:
        def create(self, **kwargs):
            return _fake_response(tool_calls=_fake_tool_call('{"move":"diagonal"}'))

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    assert agent.next_move((1, 2, 3, 4, 5, 6, 7, 0, 8)) == ""
    assert agent.last_response_metadata["tool_name"] == "submit_move"
    assert agent.last_response_metadata["status"] == "malformed"


def test_deepseek_provider_error_is_not_relabelled_as_malformed() -> None:
    class Completions:
        def create(self, **kwargs):
            raise RuntimeError("provider unavailable")

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    agent = DeepSeekAgent(api_key="not-used", client=client, thinking=False)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        agent.next_move((1, 2, 3, 4, 5, 6, 7, 0, 8))
    assert agent.last_response_metadata["status"] == "api_error"


def test_deepseek_adapter_starts_a_fresh_request_each_turn() -> None:
    class Completions:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            response = type("Response", (), {})()
            response.choices = [
                type("Choice", (), {"message": type("Message", (), {"content": "<move>left</move>"})()})()
            ]
            return response

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    agent = DeepSeekAgent(api_key="not-used", client=client)
    agent.next_move((1, 2, 3, 4, 5, 6, 7, 0, 8))
    agent.next_move((1, 2, 3, 4, 5, 6, 0, 7, 8))
    assert len(completions.calls) == 2
    assert len(completions.calls[0]["messages"]) == 2
    assert completions.calls[0]["messages"][1] != completions.calls[1]["messages"][1]
    assert all(len(call["messages"]) == 2 for call in completions.calls)


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


def test_solved_episode_uses_tile_move_convention() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, SequenceAgent(["<move>left</move>"]))
    assert result.outcome == "solved"
    assert result.reward == 1.0
    assert result.moves_taken == 1
    assert result.final_board == (1, 2, 3, 4, 5, 6, 7, 8, 0)


def test_illegal_move_ends_immediately_with_negative_reward() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, SequenceAgent(["<move>up</move>"]))
    assert result.outcome == "illegal"
    assert result.reward == -0.1
    assert result.moves_taken == 0
    assert len(result.steps) == 1


def test_malformed_response_ends_immediately_with_negative_reward() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, SequenceAgent(["I choose left"]))
    assert result.outcome == "malformed"
    assert result.reward == -0.1
    assert result.moves_taken == 0


def test_truncated_response_is_reported_separately() -> None:
    class TruncatedAgent:
        last_response_metadata = {"truncated": True}

        def next_move(self, board: tuple[int, ...]) -> str:
            return ""

    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, TruncatedAgent())
    assert result.outcome == "truncated"
    assert result.steps[0].status == "truncated"
    assert result.reward == -0.1
    assert evaluate([task], TruncatedAgent()).summary()["truncated"] == 1


def test_valid_unsolved_trajectory_at_limit_gets_zero() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate_episode(task, SequenceAgent(["<move>right</move>"]), max_turns=1)
    assert result.outcome == "timeout"
    assert result.reward == 0.0
    assert result.moves_taken == 1


def test_turn_limit_cannot_exceed_120() -> None:
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    with pytest.raises(ValueError, match="between 1 and 120"):
        evaluate_episode(task, SequenceAgent([]), max_turns=121)


def test_harness_runs_ten_huggingface_dataset_examples(monkeypatch) -> None:
    from evaluation import dataset as dataset_module

    rows = [
        {
            "board": [1, 2, 3, 4, 5, 6, 7, 0, 8],
            "optimal_moves": ["left"],
            "optimal_length": 1,
        }
    ] * 10
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *args, **kwargs: rows)
    tasks = load_examples(limit=10)

    # Use a fresh oracle that derives the next move from the current board by BFS;
    # this makes the smoke test independent of any model or credentials.
    from puzzle3.solver import solve

    class BfsOracle:
        def next_move(self, board: tuple[int, ...]) -> str:
            return f"<move>{solve(board)[0]}</move>"

    result = evaluate(tasks, BfsOracle())
    assert len(result.episodes) == 10
    assert result.summary()["solved"] == 10
    assert result.summary()["mean_reward"] == 1.0
