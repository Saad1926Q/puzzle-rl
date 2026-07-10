# smth-rl

The fifteen puzzle problem: you start out from some arbitrary configuration of 15 numbers placed on a board with one empty slot, and at each step you move one of the adjacent numbers into the empty slot, until the puzzle is solved.

In this repo, I want to try and teach an LLM to play the 15-puzzle.

The rough plan is similar to the Wordle example in Prime RL. First, I'll generate SFT data using DeepSeek on solved 15-puzzle traces. The goal of SFT is not necessarily to teach optimal solving, but to teach the model the move semantics, the board transition pattern, and the rough mental model of how legal moves lead to a solved state.

After that, I'll run RL on top of the SFT model. In RL, I want to push the model toward better solutions, and I'll also try adding a reward term that penalizes extra steps relative to an optimal solution.

## Setup

Install dependencies:

```bash
uv sync
uv pip install torch --torch-backend=auto
```

Create a `.env` file with your DeepSeek API key:

```bash
DEEPSEEK_API_KEY=your_key_here
```

## SFT

Right now, the mental model is: first do SFT to teach the model the basic format of the task, what each move means, and what a valid 15-puzzle solution trajectory looks like. The expectation is not that SFT alone will suddenly make the model good at solving the puzzle. It may help on simpler cases, but the main purpose is to give the model the right behavioral prior and action format. Then RL can build on top of that and actually teach the model how to solve harder puzzles over time.

SFT generation flow:

1. Generate candidate puzzles along with solver-verified optimal solutions.
2. Replay each optimal solution and generate a short rationale for every move using DeepSeek.
3. Export the rationale-annotated trajectories into final SFT chat-format data using `<think>` and `<move>`.
