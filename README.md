# smth-rl

The fifteen puzzle problem: you start out from some arbitrary configuration of 15 numbers placed on a board with one empty slot, and at each step you move one of the adjacent numbers into the empty slot, until the puzzle is solved.

In this repo, I want to try and teach an LLM to play the 15-puzzle.

The rough plan is similar to the Wordle example in Prime RL. First, I'll generate SFT data using DeepSeek on solved 15-puzzle traces. The goal of SFT is not necessarily to teach optimal solving, but to teach the model the move semantics, the board transition pattern, and the rough mental model of how legal moves lead to a solved state.

After that, I'll run RL on top of the SFT model. In RL, I want to push the model toward better solutions, and I'll also try adding a reward term that penalizes extra steps relative to an optimal solution.
