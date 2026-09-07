"""Provider defaults and worker-local evaluation agent factories."""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from typing import Any, Callable

from evaluation.clients.crof import CrofAgent
from evaluation.clients.deepseek import DeepSeekAgent
from evaluation.clients.glm import GLMAgent
from evaluation.clients.openai_client import OpenAIAgent
from evaluation.clients.openrouter import OpenRouterAgent
from evaluation.clients.qwen import QwenAgent
from evaluation.constants import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_CROF_API_KEY_ENV,
    DEFAULT_CROF_BASE_URL,
    DEFAULT_CROF_MODEL,
    DEFAULT_GLM_API_KEY_ENV,
    DEFAULT_GLM_BASE_URL,
    DEFAULT_GLM_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_API_KEY_ENV,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENROUTER_API_KEY_ENV,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_THINKING,
)
from evaluation.protocol import PuzzleAgent, get_api_key


type ProviderAgent = (
    CrofAgent | DeepSeekAgent | GLMAgent | OpenAIAgent | OpenRouterAgent | QwenAgent
)


@dataclass(frozen=True)
class ProviderConfig:
    agent_class: type[ProviderAgent]
    default_model: str | None
    default_base_url: str
    default_api_key_env: str | None
    default_output: str
    default_thinking: bool


PROVIDERS = {
    "crof": ProviderConfig(
        CrofAgent,
        DEFAULT_CROF_MODEL,
        DEFAULT_CROF_BASE_URL,
        DEFAULT_CROF_API_KEY_ENV,
        "results_8puzzle_crof_glm_5_3_flash.json",
        DEFAULT_THINKING,
    ),
    "deepseek": ProviderConfig(
        DeepSeekAgent,
        DEFAULT_MODEL,
        DEFAULT_BASE_URL,
        DEFAULT_API_KEY_ENV,
        "results_8puzzle_deepseek_v4_flash.json",
        DEFAULT_THINKING,
    ),
    "glm": ProviderConfig(
        GLMAgent,
        DEFAULT_GLM_MODEL,
        DEFAULT_GLM_BASE_URL,
        DEFAULT_GLM_API_KEY_ENV,
        "results_8puzzle_glm_4_5_air.json",
        DEFAULT_THINKING,
    ),
    "openai": ProviderConfig(
        OpenAIAgent,
        DEFAULT_OPENAI_MODEL,
        DEFAULT_OPENAI_BASE_URL,
        DEFAULT_OPENAI_API_KEY_ENV,
        "results_8puzzle_openai.json",
        DEFAULT_THINKING,
    ),
    "openrouter": ProviderConfig(
        OpenRouterAgent,
        None,
        DEFAULT_OPENROUTER_BASE_URL,
        DEFAULT_OPENROUTER_API_KEY_ENV,
        "results_8puzzle_openrouter.json",
        DEFAULT_THINKING,
    ),
    "qwen": ProviderConfig(
        QwenAgent,
        DEFAULT_QWEN_MODEL,
        DEFAULT_QWEN_BASE_URL,
        None,
        "results_8puzzle_qwen3_5_0_8b.json",
        False,
    ),
}


def resolve_provider_args(args: argparse.Namespace) -> None:
    """Fill provider defaults and reject missing provider-specific settings."""
    provider = PROVIDERS[args.provider]
    args.model = args.model or provider.default_model
    args.base_url = args.base_url or provider.default_base_url
    args.api_key_env = args.api_key_env or provider.default_api_key_env
    if args.provider == "openrouter" and args.model is None:
        raise ValueError("--model is required for --provider openrouter")
    if args.thinking is None:
        args.thinking = provider.default_thinking
    if args.reasoning_effort is None and args.provider != "openrouter":
        args.reasoning_effort = DEFAULT_REASONING_EFFORT


def create_agent_factory(args: argparse.Namespace) -> Callable[[], PuzzleAgent]:
    """Create one agent per evaluation worker, reusing it within that worker."""
    provider = PROVIDERS[args.provider]
    api_key = (
        get_api_key(args.api_key_env, args.dotenv)
        if args.api_key_env is not None
        else "not-required"
    )
    worker_local = threading.local()

    def agent_factory() -> PuzzleAgent:
        agent = getattr(worker_local, "agent", None)
        if agent is None:
            agent_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "model": args.model,
                "base_url": args.base_url,
                "thinking": args.thinking,
                "reasoning_effort": args.reasoning_effort,
                "max_tokens": args.max_tokens,
            }
            if args.provider == "qwen":
                agent_kwargs.update(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    presence_penalty=args.presence_penalty,
                    repetition_penalty=args.repetition_penalty,
                )
            elif args.provider == "openrouter":
                agent_kwargs.update(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    upstream_providers=args.openrouter_upstream,
                    allow_fallbacks=args.openrouter_allow_fallbacks,
                    require_parameters=not args.openrouter_relax_parameters,
                    data_collection=args.openrouter_data_collection,
                    distillable_only=args.openrouter_distillable_only,
                    quantizations=args.openrouter_quantization,
                )
            agent = provider.agent_class(**agent_kwargs)
            worker_local.agent = agent
        return agent

    return agent_factory
