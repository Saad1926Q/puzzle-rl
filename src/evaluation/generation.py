"""Shared exact-state enumeration and evaluation dataset serialization."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from datasets import Dataset

from evaluation.constants import ACTION_INTERFACE
from puzzle3.board import Board, GOAL, TileAction, adjacent_tiles, slide_tile


def enumerate_from_goal() -> dict[Board, list[TileAction]]:
    """Return every reachable 8-puzzle state with an optimal path from GOAL."""

    queue = deque([GOAL])
    paths: dict[Board, list[TileAction]] = {GOAL: []}

    while queue:
        board = queue.popleft()
        for tile in adjacent_tiles(board):
            next_board = slide_tile(board, tile)
            if next_board in paths:
                continue
            paths[next_board] = [*paths[board], tile]
            queue.append(next_board)

    return paths


def solution_from_goal_path(path_from_goal: list[TileAction]) -> list[TileAction]:
    """Reverse a GOAL-to-board path into a board-to-GOAL solution."""

    return list(reversed(path_from_goal))


def make_eval_record(
    board: Board, bucket: str, path_from_goal: list[TileAction]
) -> dict[str, Any]:
    """Build one validated-schema evaluation record."""

    optimal_actions = solution_from_goal_path(path_from_goal)
    return {
        "board": list(board),
        "scramble_depth": len(optimal_actions),
        "bucket": bucket,
        "action_interface": ACTION_INTERFACE,
        "optimal_actions": optimal_actions,
        "optimal_length": len(optimal_actions),
    }


def write_eval_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def write_eval_parquet(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_path))
