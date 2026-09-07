from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from evaluation.vllm import RuntimeLoRAController, VLLMLoRAError


class FakeResponse:
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return b"ok"


def test_runtime_lora_controller_loads_and_unloads_adapter(tmp_path: Path) -> None:
    requests: list[tuple[str, dict[str, object]]] = []

    def opener(request: object, *, timeout: float) -> FakeResponse:
        del timeout
        url = request.full_url
        payload = json.loads(request.data.decode("utf-8"))
        requests.append((url, payload))
        return FakeResponse()

    controller = RuntimeLoRAController("http://localhost:8000/v1", opener=opener)
    controller.load("checkpoint-20", tmp_path)
    controller.unload("checkpoint-20")

    assert requests == [
        (
            "http://localhost:8000/v1/unload_lora_adapter",
            {"lora_name": "checkpoint-20"},
        ),
        (
            "http://localhost:8000/v1/load_lora_adapter",
            {"lora_name": "checkpoint-20", "lora_path": str(tmp_path.resolve())},
        ),
        (
            "http://localhost:8000/v1/unload_lora_adapter",
            {"lora_name": "checkpoint-20"},
        ),
    ]


def test_runtime_lora_controller_ignores_missing_adapter_on_load() -> None:
    def opener(request: object, *, timeout: float) -> object:
        del request, timeout
        raise HTTPError(
            "http://localhost:8000/v1/unload_lora_adapter",
            404,
            "missing",
            {},
            None,
        )

    controller = RuntimeLoRAController("http://localhost:8000/v1", opener=opener)
    controller.unload("checkpoint-20", ignore_missing=True)


def test_runtime_lora_controller_reports_server_errors() -> None:
    def opener(request: object, *, timeout: float) -> object:
        del request, timeout
        raise HTTPError(
            "http://localhost:8000/v1/load_lora_adapter",
            500,
            "broken",
            {},
            None,
        )

    controller = RuntimeLoRAController("http://localhost:8000/v1", opener=opener)
    with pytest.raises(VLLMLoRAError, match="HTTP 500"):
        controller.load("checkpoint-20", Path("checkpoint-20"))
