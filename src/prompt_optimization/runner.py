"""Run GEPA against the existing OpenRouter 8-puzzle evaluator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gepa

from evaluation.clients.openrouter import OpenRouterAgent
from evaluation.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TURNS,
    DEFAULT_OPENROUTER_API_KEY_ENV,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_THINKING,
    SEED_STRATEGY_PROMPT,
)
from evaluation.protocol import get_api_key
from prompt_optimization.adapter import PuzzleGEPAAdapter, STRATEGY_COMPONENT
from prompt_optimization.dataset import GEPASplits


@dataclass(frozen=True)
class OpenRouterRolloutConfig:
    """Reproducible rollout settings shared by every GEPA candidate."""

    model: str
    api_key_env: str = DEFAULT_OPENROUTER_API_KEY_ENV
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    thinking: bool = DEFAULT_THINKING
    reasoning_effort: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 1.0
    top_p: float = 1.0
    upstream_providers: tuple[str, ...] = ()
    allow_fallbacks: bool = False
    data_collection: str = "deny"
    distillable_only: bool = False
    quantizations: tuple[str, ...] = ()
    max_turns: int = DEFAULT_MAX_TURNS
    keep_history: bool = True
    keep_reasoning: bool = True


def run_gepa_optimization(
    *,
    splits: GEPASplits,
    rollout: OpenRouterRolloutConfig,
    reflection_model: str,
    output_dir: Path,
    max_metric_calls: int,
    reflection_minibatch_size: int,
    seed: int,
) -> dict[str, object]:
    """Optimize exactly one strategy prompt and persist GEPA's complete result."""

    if max_metric_calls <= 0 or reflection_minibatch_size <= 0:
        raise ValueError("GEPA budgets must be positive")
    if rollout.keep_reasoning and not rollout.keep_history:
        raise ValueError("keep_reasoning requires keep_history")

    api_key = get_api_key(rollout.api_key_env)
    output_dir.mkdir(parents=True, exist_ok=True)

    def agent_factory(system_prompt: str) -> OpenRouterAgent:
        return OpenRouterAgent(
            api_key=api_key,
            model=rollout.model,
            base_url=rollout.base_url,
            thinking=rollout.thinking,
            reasoning_effort=rollout.reasoning_effort,
            max_tokens=rollout.max_tokens,
            temperature=rollout.temperature,
            top_p=rollout.top_p,
            upstream_providers=rollout.upstream_providers,
            allow_fallbacks=rollout.allow_fallbacks,
            data_collection=rollout.data_collection,
            distillable_only=rollout.distillable_only,
            quantizations=rollout.quantizations,
            system_prompt=system_prompt,
        )

    adapter = PuzzleGEPAAdapter(
        agent_factory=agent_factory,
        max_turns=rollout.max_turns,
        keep_history=rollout.keep_history,
        keep_reasoning=rollout.keep_reasoning,
    )
    config = {
        "rollout": asdict(rollout),
        "reflection_model": reflection_model,
        "max_metric_calls": max_metric_calls,
        "reflection_minibatch_size": reflection_minibatch_size,
        "seed": seed,
        "split_sizes": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "seed_strategy.txt").write_text(
        SEED_STRATEGY_PROMPT + "\n", encoding="utf-8"
    )

    result = gepa.optimize(
        seed_candidate={STRATEGY_COMPONENT: SEED_STRATEGY_PROMPT},
        trainset=splits.train,
        valset=splits.validation,
        adapter=adapter,
        reflection_lm=reflection_model,
        reflection_lm_kwargs={"temperature": 1.0, "max_tokens": 16_000},
        candidate_selection_strategy="pareto",
        reflection_minibatch_size=reflection_minibatch_size,
        max_metric_calls=max_metric_calls,
        run_dir=str(output_dir / "gepa"),
        seed=seed,
        use_merge=False,
        track_best_outputs=False,
        cache_evaluation=False,
    )
    best_candidate = result.best_candidate
    assert isinstance(best_candidate, dict)
    best_strategy = best_candidate[STRATEGY_COMPONENT]
    (output_dir / "best_strategy.txt").write_text(best_strategy + "\n", encoding="utf-8")
    result_dict = result.to_dict()
    (output_dir / "result.json").write_text(
        json.dumps(result_dict, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "best_strategy": best_strategy,
        "best_validation_reward": result.val_aggregate_scores[result.best_idx],
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
    }
