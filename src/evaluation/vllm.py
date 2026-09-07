"""Small control client for vLLM's runtime LoRA endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class VLLMLoRAError(RuntimeError):
    """Raised when vLLM cannot load or unload a LoRA adapter."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RuntimeLoRAController:
    """Load one local LoRA adapter at a time into a running vLLM server."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._opener = opener

    def load(self, name: str, path: Path) -> None:
        """Replace any existing adapter with ``name`` from ``path``."""

        self.unload(name, ignore_missing=True)
        self._post(
            "load_lora_adapter",
            {"lora_name": name, "lora_path": str(path.resolve())},
        )

    def unload(self, name: str, *, ignore_missing: bool = False) -> None:
        """Unload ``name`` from vLLM."""

        try:
            self._post("unload_lora_adapter", {"lora_name": name})
        except VLLMLoRAError as exc:
            if ignore_missing and exc.status_code in {400, 404}:
                return
            raise

    def _post(self, endpoint: str, payload: dict[str, Any]) -> None:
        request = Request(
            f"{self.base_url}/{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                response.read()
        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            raise VLLMLoRAError(
                f"vLLM {endpoint} failed with HTTP {exc.code}: {detail}",
                status_code=exc.code,
            ) from exc
        except URLError as exc:
            raise VLLMLoRAError(
                f"could not reach vLLM at {self.base_url}: {exc.reason}"
            ) from exc
        except OSError as exc:
            raise VLLMLoRAError(
                f"could not reach vLLM at {self.base_url}: {exc}"
            ) from exc
