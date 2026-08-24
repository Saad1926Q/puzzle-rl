# Experiment Log

## 2026-08-23 to 2026-08-24 - Base-model selection

Ran baseline evaluations to select a Qwen3.5 checkpoint for 8-puzzle RL. Evaluations used the native `slide_tile` tool, exact-distance rewards, repeated rollouts, and both thinking and non-thinking inference.

### Experiments

- **Qwen3.5-0.8B:** non-thinking runs at 128 and 256 tokens; thinking runs at 1,024 and 2,048 tokens.
- **Qwen3.5-2B:** non-thinking at 128 tokens; thinking at 1,024 and 4,096 tokens.
- **Qwen3.5-4B:** non-thinking on easy, medium, and the full 45-puzzle dataset at 128 tokens; thinking smoke test at 4,096 tokens.
- Main evaluations used eight rollouts per puzzle.

Early runs were invalidated because the vLLM thinking flag was passed incorrectly and the prompt did not explicitly state the goal board. Both issues were corrected before the retained comparisons.

### Current result

Qwen3.5-4B non-thinking is the best base-model candidate. On the full dataset (45 puzzles, eight rollouts each; 360 episodes), it achieved:

- 41 solved episodes (11.4%)
- 24.4% pass@8
- 87.5% illegal outcomes
- 0 malformed outcomes
- 3 truncated and 1 timed-out episode

Performance by difficulty:

| Depth | Solved rate | Pass@8 |
| ----- | ----------: | -----: |
| 1–5   |       32.5% |  66.7% |
| 6–15  |        1.7% |   6.7% |
| 16–31 |          0% |     0% |

Observed per-depth outcomes were strongest at depths 3–4, weak at depths 5–6, and zero from the sampled depths 10–31. The 0.8B and 2B checkpoints were substantially weaker, while thinking mode was slower and dominated by malformed or truncated responses.

### Next step

Run a denser held-out evaluation across depths 3–10 before choosing the RL mixture or adding targeted rejection-sampling/solver-generated SFT.
