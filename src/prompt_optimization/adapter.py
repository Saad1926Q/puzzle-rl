"""GEPA adapter that evaluates strategy prompts through the existing environment."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from gepa.core.adapter import EvaluationBatch

from evaluation.dataset import PuzzleExample
from prompt_optimization.eval.constants import build_candidate_system_prompt
from prompt_optimization.eval.evaluator import EpisodeResult, evaluate_episode
from prompt_optimization.eval.protocol import PuzzleAgent
from prompt_optimization.feedback import episode_reflection_record

STRATEGY_COMPONENT = "strategy_prompt"
AgentFactory = Callable[[str], PuzzleAgent]


class PuzzleGEPAAdapter:
    """Evaluate GEPA candidates using the authoritative multi-turn evaluator."""

    propose_new_texts = None

    def __init__(
        self,
        *,
        agent_factory: AgentFactory,
        max_turns: int,
        keep_history: bool,
        keep_reasoning: bool,
        parallelism: int = 8,
    ) -> None:
        if parallelism <= 0:
            raise ValueError("parallelism must be positive")
        self.agent_factory = agent_factory
        self.max_turns = max_turns
        self.keep_history = keep_history
        self.keep_reasoning = keep_reasoning
        self.parallelism = parallelism

    @staticmethod
    def system_prompt(strategy_prompt: str) -> str:
        """Build the only prompt text supplied to a rollout agent."""

        return build_candidate_system_prompt(strategy_prompt)

    def evaluate(
        self,
        batch: list[PuzzleExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[EpisodeResult, EpisodeResult]:
        """Run one authoritative episode per puzzle and return its native reward."""

        try:
            strategy_prompt = candidate[STRATEGY_COMPONENT]
        except KeyError as exc:
            raise ValueError(f"candidate must contain {STRATEGY_COMPONENT!r}") from exc
        prompt = self.system_prompt(strategy_prompt)

        def run_one(example: PuzzleExample) -> EpisodeResult:
            return evaluate_episode(
                example,
                self.agent_factory(prompt),
                max_turns=self.max_turns,
                keep_history=self.keep_history,
                keep_reasoning=self.keep_reasoning,
            )

        if not batch:
            return EvaluationBatch(
                outputs=[],
                scores=[],
                trajectories=[] if capture_traces else None,
                num_metric_calls=0,
            )

        episodes: list[EpisodeResult | None] = [None] * len(batch)
        workers = min(self.parallelism, len(batch))
        with (
            ThreadPoolExecutor(max_workers=workers) as executor,
            tqdm(total=len(batch), desc="Puzzle rollouts", unit="episode", leave=False) as progress,
        ):
            future_indices: dict[Future[EpisodeResult], int] = {
                executor.submit(run_one, example): index
                for index, example in enumerate(batch)
            }
            for future in as_completed(future_indices):
                episodes[future_indices[future]] = future.result()
                progress.update(1)

        completed = [episode for episode in episodes if episode is not None]
        if len(completed) != len(batch):
            raise RuntimeError("not all puzzle rollouts produced a result")
        return EvaluationBatch(
            outputs=completed,
            scores=[episode.reward for episode in completed],
            trajectories=completed if capture_traces else None,
            num_metric_calls=len(completed),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[EpisodeResult, EpisodeResult],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Prepare concise, strategy-relevant evidence for GEPA's reflection LM."""

        del candidate
        if eval_batch.trajectories is None:
            raise ValueError("reflection requires captured trajectories")
        unknown = set(components_to_update) - {STRATEGY_COMPONENT}
        if unknown:
            raise ValueError(f"unknown GEPA components requested: {sorted(unknown)}")
        records = [
            episode_reflection_record(episode)
            for episode in eval_batch.trajectories
            if episode.outcome != "api_error"
        ]
        return {component: records for component in components_to_update}
