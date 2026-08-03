#!/usr/bin/env bash
# Usage: scripts/train_destroy.sh INSTANCE_ID WANDB_PROJECT TRAIN_ARGS...
# Example: scripts/train_destroy.sh 12345 fifteen-puzzle configs/trl/train.yaml --push-to-hub user/repo
set -euo pipefail

INSTANCE_ID="$1"
WANDB_PROJECT="$2"
shift 2

export WANDB_PROJECT
export TRL_EXPERIMENTAL_SILENCE="${TRL_EXPERIMENTAL_SILENCE:-1}"

uv run python -m rl.train "$@"
vastai destroy instance "$INSTANCE_ID"
