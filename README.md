# smth-rl

A tiny sliding-puzzle RL/eval sandbox focused first on the **3x3 / 8-puzzle**.

This project started as a small experiment to teach language models how to solve the 15-puzzle. Here is a bit about how that went. I tried training a small Qwen3-1.7B model with GRPO in a multi-turn environment: at each step, the model selected one move and tried to reach the solved state. Rather than sending the entire history back to the model, I sent only the current board, since each puzzle state is a self-contained problem and including a long trajectory would unnecessarily grow the context. The environment allows up to 45 turns.

The reward was `0.8 + 0.2 × min(optimal_length / moves_taken, 1)` for a solved puzzle, `-0.1` for an illegal move or malformed response, and `0` for a valid trajectory that reached the turn limit without solving. Before GRPO, I ran a 40-step SFT warm-up using synthetic trajectories generated with DeepSeek V4 Flash. For RL, I used a Gaussian easy-to-hard curriculum over four buckets: trivial, easy, medium, and hard. I initially thought the project was wrapped once RL training finished, but evaluation exposed a frustrating failure: illegal moves decreased, while the number of solved puzzles stayed roughly the same. The model had mostly learned to make valid but random moves until the 45-turn limit, avoiding the negative reward without actually learning how to solve the puzzle.

I then took a step back to understand how to design the experimentation properly: how to establish meaningful baselines, structure environments and rollouts, define rewards, and generally run good RL experiments. The HANABI blog post by nphard was especially useful for thinking through these questions. That led to starting with the smaller 8-puzzle, where the core mechanics are easier to validate and failure modes are cheaper to inspect before returning to the 15-puzzle.

## Why 3x3 first?

The 15-puzzle has a huge state space and long optimal trajectories, so it is hard to tell whether RL is failing because of basic action semantics, looping, weak local heuristics, or genuinely hard long-horizon planning.

Inspired by the move from full Hanabi to Tiny Hanabi described in nphard's HANABI blog, the 3x3 / 8-puzzle is a better first microscope:

- only 181,440 solvable states;
- max optimal solution length is 31 moves;
- exact optimal labels are easy to generate by BFS;
- rollout traces are short enough to inspect directly;
- learned skills such as move semantics, anti-loop control, and local heuristic progress may later transfer to 4x4.

The current research loop is:

```text
baseline evals → rollout inspection → anti-loop/progress rewards → later test transfer to 4x4
```

## Setup

Install dependencies and provide the API key in `.env` (or export it):

```bash
uv sync
uv pip install torch --torch-backend=auto
```

```dotenv
DEEPSEEK_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## Model evaluation

The harness loads puzzles directly from the Hugging Face `eval` split of
`saad1926q/8-puzzle` (the full split by default). Each turn sends the selected model only
the current board and requests one strict `submit_move` tool call. Thinking mode is
enabled by default with a 4096-token budget; use `--no-thinking` to force the tool call
without reasoning.

Run DeepSeek using the key in `.env`:

```bash
uv run python scripts/run_eval_8puzzle.py \
  --provider deepseek \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com/beta \
  --max-tokens 4096 \
  --output eval/results_8puzzle_deepseek_v4_flash.json
```

Run an OpenAI reasoning model with the same board-only protocol:

```bash
uv run python scripts/run_eval_8puzzle.py \
  --provider openai \
  --model gpt-5.4 \
  --reasoning-effort low \
  --max-tokens 8192 \
  --output eval/results_openai_gpt54.json
```

OpenAI uses `OPENAI_API_KEY` by default and the adapter translates `--max-tokens`
to OpenAI's `max_output_tokens` in the Responses API. Override the endpoint or
environment variable with `--base-url` and `--api-key-env` when using an
OpenAI-compatible service.

The full eval split is used by default. Use `--num-examples N` to evaluate only the
first `N` rows. Use `--num-rollouts K` for `K` independent attempts per puzzle; the
summary then reports rollout metrics and exact `pass@k` (the fraction of puzzles solved
by at least one rollout). Use `--parallelism N` to run independent episodes concurrently
while keeping turns within each episode sequential. Use `--provider openai` or
`--provider deepseek` to select the adapter. Other useful options include
`--offset`, `--max-turns` (up to 45), `--reasoning-effort`, and `--max-tokens`.

The output JSON contains metadata and aggregate metrics by default. To save complete
per-example trajectories separately, add `--save-trajectories`:

```bash
uv run python scripts/run_eval_8puzzle.py \
  --num-examples 10 \
  --num-rollouts 3 \
  --save-trajectories \
  --output eval/results_8puzzle_deepseek_v4_flash.json
```

This writes the compact summary to the requested path and full traces to a sibling
`.trajectories.json` file. Trajectory steps include boards, moves, reasoning, tool calls,
usage, finish reasons, and truncation status.

Reported metrics include solved, illegal, malformed, truncated, and timeout counts/rates,
mean reward, and mean move counts. The solved reward is
`0.8 + 0.2 × min(optimal_length / moves_taken, 1)`; illegal or malformed responses get
`-0.1`, and valid unsolved trajectories get `0` at the turn limit.
