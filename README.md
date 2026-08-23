<h1 align="center">puzzle-rl</h1>

<p align="center">
A small environment for evaluating and training language models on the 8-puzzle
</p>

---

## Motivation

I am trying to teach language models to solve 2D sliding puzzles. I started with the 4x4 / 15-puzzle, but its search space is enormous and difficult puzzles can require around 80 moves to solve optimally. That made it hard to run controlled experiments and understand why a model failed.

I therefore moved to the smaller 3x3 / 8-puzzle first. Its simpler state space and shorter solutions make it easier to inspect trajectories, validate the environment and rewards, and determine whether a model is actually learning to plan.

## What This Repo Is About

This repository provides:

- an 8-puzzle environment with deterministic transitions;
- a tile-based tool interface for language models;
- exact solution distances and optimal paths;
- a fixed, mixed-difficulty evaluation set;
- adapters for DeepSeek, GLM, and OpenAI models;
- multi-rollout evaluation with separate trajectory files;
- rewards suitable for later RL experiments.

Each model request contains only the current board. Previous messages and reasoning are not included because the board already contains the complete environment state.

## Action Interface

The model selects the numbered tile that should slide into the blank:

```json
{
  "tile": 8
}
```

The action is submitted through one strict tool call:

```text
slide_tile(tile=8)
```

The selected tile must be numbered `1` through `8` and adjacent to the blank (`0`). Selecting a non-adjacent tile ends the episode as illegal. Missing, invalid, or multiple tool calls are treated as malformed responses.

## Reward

Let `D(s)` be the exact number of moves required to solve board state `s`. Every valid transition receives a small reward for reducing that distance:

```text
progress reward = 0.25 × (D_before - D_after) / 31
```

Moving closer is positive, moving farther away is negative, and preserving the same distance gives zero.

A solved episode also receives:

```text
0.8 + 0.2 × min(optimal_length / moves_taken, 1)
```

The remaining outcomes are handled as follows:

| Outcome | Reward |
| --- | ---: |
| Illegal action | `-1.0` |
| Malformed response | `-1.0` |
| Truncated response | `-1.0` |
| Timeout | accumulated progress reward `- 0.25` |

Training and evaluation use the same deterministic reward calculation. Serialized steps record the total reward, progress reward, and terminal reward separately.

## Evaluation Set

The default evaluation set is [`saad1926q/8-puzzle`](https://huggingface.co/datasets/saad1926q/8-puzzle). Its `eval` split contains 45 fixed puzzles:

- 15 easy puzzles;
- 15 medium puzzles;
- 15 hard puzzles.

The repository can also generate equivalent local JSONL and Parquet files:

```bash
uv run python data/generate_eval_data_3x3.py
```

The generator exhaustively searches the solvable state space, groups puzzles by difficulty, and selects reproducible state quantiles within each group.

## Metrics and Trajectories

Evaluation reports:

- solved, illegal, malformed, truncated, and timeout rates;
- mean episode reward;
- mean moves taken;
- solution efficiency;
- `pass@k` when multiple rollouts are used.

Compact metrics and complete trajectories are stored separately. A trajectory includes every board state, legal tile set, submitted tile, reward component, tool call, finish reason, token usage, and available reasoning metadata.

## Running the Project

This project requires Python 3.13 or newer and uses [uv](https://docs.astral.sh/uv/).

Install the dependencies:

```bash
uv sync
```

Provide the relevant API key through the environment or a local `.env` file:

```dotenv
DEEPSEEK_API_KEY=your_key_here
ZAI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

Run an evaluation:

```bash
uv run python scripts/run_eval_8puzzle.py \
    --provider glm \
    --model glm-4.7 \
    --dataset data/eval_puzzles_3x3_45.jsonl \
    --num-rollouts 3 \
    --parallelism 3 \
    --thinking \
    --max-tokens 4096 \
    --save-trajectories \
    --output eval/glm-4.7.json
```

The command writes:

```text
eval/glm-4.7.json
eval/glm-4.7.trajectories.json
```

Use another provider with:

```text
--provider deepseek
--provider glm
--provider openai
```

Useful options include:

```text
--num-examples N     evaluate only N puzzles
--offset N           start from a later dataset row
--num-rollouts K     run K attempts per puzzle
--parallelism N      run independent episodes concurrently
--max-turns N        limit each episode to at most 45 actions
--no-thinking        disable model reasoning when supported
--reasoning-effort   choose low, medium, high, or max
--max-tokens N       set the response token budget
```

Run the tests with:

```bash
uv run pytest -q
```
