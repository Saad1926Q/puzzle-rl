from __future__ import annotations

from dataclasses import dataclass
import json

import pytest
from datasets import Dataset

from evaluation.dataset import DatasetError, PuzzleExample, load_examples
from evaluation.evaluator import evaluate
from evaluation.protocol import get_api_key
from evaluation.reporting import metadata, write_evaluation_artifacts


@dataclass
class SequenceAgent:
    responses: list[str]

    def next_action(self, board: tuple[int, ...]) -> str:
        return self.responses.pop(0)

def example(board: tuple[int, ...], optimal_length: int = 1) -> PuzzleExample:
    return PuzzleExample("test", board, tuple(), optimal_length, {})

def test_qwen_cli_uses_local_defaults_and_needs_no_api_key() -> None:
    import runpy

    runner = runpy.run_path("scripts/run_eval_8puzzle.py")
    xhigh_args = runner["build_parser"]().parse_args(
        ["--provider", "qwen", "--reasoning-effort", "xhigh"]
    )
    history_args = runner["build_parser"]().parse_args(
        ["--provider", "qwen", "--history"]
    )
    no_history_args = runner["build_parser"]().parse_args(["--provider", "qwen"])
    args = runner["build_parser"]().parse_args(["--provider", "qwen"])
    settings = runner["ProviderSettings"].from_args(args)

    assert settings.model == "Qwen/Qwen3.5-0.8B"
    assert settings.board_representation == "grid"
    assert xhigh_args.reasoning_effort == "xhigh"
    assert settings.api_key_env is None
    assert settings.thinking is False
    assert args.history is False
    assert no_history_args.history is False
    assert history_args.history is True
    custom_args = runner["build_parser"]().parse_args(
        ["--provider", "qwen", "--board-representation", "markdown"]
    )
    assert runner["ProviderSettings"].from_args(custom_args).board_representation == (
        "markdown"
    )
    crof_args = runner["build_parser"]().parse_args(["--provider", "crof"])
    crof_settings = runner["ProviderSettings"].from_args(crof_args)
    assert crof_settings.model == "glm-5.3-flash"
    assert crof_settings.base_url == "https://crof.ai/v1"
    assert crof_settings.api_key_env == "CROF_API_KEY"


def test_checkpoint_episode_round_trip(tmp_path) -> None:
    import runpy

    runner = runpy.run_path("scripts/run_eval_8puzzle.py")
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    result = evaluate([task], SequenceAgent(['{"tile": 8}']))
    settings = {key: None for key in runner["CHECKPOINT_SETTINGS"]}
    settings["num_rollouts"] = 1
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "metadata",
                        "version": runner["CHECKPOINT_VERSION"],
                        "settings": settings,
                    }
                ),
                json.dumps(
                    {
                        "type": "episode",
                        "example_index": 0,
                        "rollout_id": 0,
                        "episode": result.episodes[0].to_dict(),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed, episodes = runner["load_checkpoint"](
        checkpoint,
        expected_settings=settings,
        examples=[task],
    )

    assert completed == {(0, 0)}
    assert episodes[0][0] == (0, 0)
    assert episodes[0][1].to_dict() == result.episodes[0].to_dict()
def test_cli_checkpoint_resume_reuses_completed_episodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import runpy
    import sys

    runner = runpy.run_path("scripts/run_eval_8puzzle.py")
    task = example((1, 2, 3, 4, 5, 6, 7, 0, 8))
    runner["main"].__globals__["load_examples"] = lambda **_kwargs: [task]
    runner["main"].__globals__["create_agent_factory"] = (
        lambda _settings, _dotenv: lambda: SequenceAgent(['{"tile": 8}'])
    )
    checkpoint = tmp_path / "checkpoint.jsonl"
    output = tmp_path / "summary.json"
    arguments = [
        "run_eval_8puzzle.py",
        "--provider",
        "qwen",
        "--num-rollouts",
        "2",
        "--checkpoint-path",
        str(checkpoint),
        "--output",
        str(output),
    ]

    monkeypatch.setattr(sys, "argv", arguments)
    runner["main"]()
    first_checkpoint_lines = checkpoint.read_text(encoding="utf-8").splitlines()
    assert len(first_checkpoint_lines) == 3

    monkeypatch.setattr(sys, "argv", [*arguments, "--resume"])
    runner["main"]()

    assert checkpoint.read_text(encoding="utf-8").splitlines() == first_checkpoint_lines
    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["num_episodes"] == 2


def test_cli_passes_agent_factory_to_evaluator(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import runpy
    import sys

    runner = runpy.run_path("scripts/run_eval_8puzzle.py")
    captured: dict[str, object] = {}
    agent_factory = object()

    class Result:
        def summary(self) -> dict[str, object]:
            return {}

    def fake_evaluate(*args, **kwargs):
        captured.update(kwargs)
        return Result()

    globals_ = runner["main"].__globals__
    globals_["load_examples"] = lambda **kwargs: []
    globals_["create_agent_factory"] = lambda settings, dotenv: agent_factory
    globals_["evaluate"] = fake_evaluate
    globals_["metadata"] = lambda *args: {}
    globals_["write_evaluation_artifacts"] = lambda *args, **kwargs: None
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval_8puzzle.py",
            "--provider",
            "qwen",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )

    runner["main"]()

    assert captured["agent_factory"] is agent_factory


def test_reporting_preserves_summary_and_trajectory_artifact_schemas(tmp_path) -> None:
    import runpy

    runner = runpy.run_path("scripts/run_eval_8puzzle.py")
    args = runner["build_parser"]().parse_args(["--provider", "qwen"])
    args.save_trajectories = True
    settings = runner["ProviderSettings"].from_args(args)
    result = evaluate(
        [example((1, 2, 3, 4, 5, 6, 7, 0, 8))],
        SequenceAgent(['{"tile": 8}']),
    )
    output = tmp_path / "summary.json"
    run_metadata = metadata(args, 1, settings)

    trajectory_path = write_evaluation_artifacts(
        output,
        run_metadata=run_metadata,
        result=result,
        save_trajectories=args.save_trajectories,
    )

    assert trajectory_path == tmp_path / "summary.trajectories.json"
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "metadata": run_metadata,
        "summary": result.summary(),
        "trajectory_file": str(trajectory_path),
    }
    assert json.loads(trajectory_path.read_text(encoding="utf-8")) == {
        "metadata": run_metadata,
        "summary": result.summary(),
        "episodes": [episode.to_dict() for episode in result.episodes],
    }

def test_openrouter_cli_requires_model_and_uses_reproducible_defaults() -> None:
    import runpy

    runner = runpy.run_path("scripts/run_eval_8puzzle.py")
    parser = runner["build_parser"]()
    args = parser.parse_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "qwen/qwen3.5-27b",
            "--openrouter-upstream",
            "together",
            "--openrouter-quantization",
            "bf16",
            "--openrouter-distillable-only",
        ]
    )
    settings = runner["ProviderSettings"].from_args(args)

    assert settings.base_url == "https://openrouter.ai/api/v1"
    assert settings.api_key_env == "OPENROUTER_API_KEY"
    assert settings.openrouter_allow_fallbacks is False
    assert settings.openrouter_data_collection == "deny"
    assert settings.openrouter_upstream == ("together",)
    assert settings.openrouter_quantizations == ("bf16",)
    assert settings.openrouter_require_parameters is True
    relaxed_args = parser.parse_args(
        [
            "--provider",
            "openrouter",
            "--model",
            "qwen/qwen3.5-27b",
            "--openrouter-relax-parameters",
        ]
    )
    assert relaxed_args.openrouter_relax_parameters is True
    assert settings.reasoning_effort is None

    missing_model = parser.parse_args(["--provider", "openrouter"])
    with pytest.raises(ValueError, match="--model is required"):
        runner["ProviderSettings"].from_args(missing_model)

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

