from __future__ import annotations

import runpy
import sys

import pytest


def load_runner() -> dict[str, object]:
    return runpy.run_path("scripts/train_rl.py")


def test_train_cli_accepts_vllm_max_model_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_rl.py", "--vllm-max-model-length", "32768"],
    )

    args = runner["parse_args"]()

    assert args.vllm_max_model_length == 32768


def test_train_cli_rejects_nonpositive_vllm_max_model_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_rl.py", "--vllm-max-model-length", "0"],
    )

    with pytest.raises(ValueError, match="vllm-max-model-length must be positive"):
        runner["main"]()


