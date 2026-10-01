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


def test_train_cli_accepts_resume_and_adapter_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_rl.py",
            "--resume-from-checkpoint",
            "hf://user/full-checkpoint",
            "--lora-adapter",
            "hf://user/adapter",
        ],
    )

    args = runner["parse_args"]()

    assert args.resume_from_checkpoint == "hf://user/full-checkpoint"
    assert args.lora_adapter == "hf://user/adapter"


def test_resume_checkpoint_requires_trainer_state(tmp_path) -> None:
    runner = load_runner()

    with pytest.raises(ValueError, match="missing trainer_state.json"):
        runner["_resolve_resume_checkpoint"](str(tmp_path))


def test_resume_checkpoint_accepts_trainer_state(tmp_path) -> None:
    runner = load_runner()
    (tmp_path / "trainer_state.json").write_text("{}")

    assert runner["_resolve_resume_checkpoint"](str(tmp_path)) == tmp_path


def test_lora_adapter_loads_base_model_without_vllm(
    tmp_path,
) -> None:
    runner = load_runner()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base, path, *, is_trainable):
            return base, path, is_trainable

    globals_ = runner["_load_lora_adapter"].__globals__
    globals_["PeftModel"] = FakePeftModel
    globals_["create_model_from_path"] = lambda model: ("base", model)

    assert runner["_load_lora_adapter"]("Qwen/base", str(adapter)) == (
        ("base", "Qwen/base"),
        str(adapter),
        True,
    )


def test_adapter_source_can_download_from_hugging_face(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    runner = load_runner()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    runner["_resolve_lora_adapter"].__globals__["snapshot_download"] = (
        lambda **_kwargs: str(adapter)
    )

    resolved = runner["_resolve_lora_adapter"]("hf://user/adapter")

    assert resolved == adapter


