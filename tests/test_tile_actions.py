from __future__ import annotations

import random
from collections import Counter

import pytest

from evaluation.dataset import load_examples
from puzzle3 import scramble as scramble_module
from puzzle3.board import GOAL, adjacent_tiles, slide_tile
from puzzle3.solver import exact_distance, solve


def test_adjacent_tiles_uses_board_connectivity_order() -> None:
    assert adjacent_tiles((1, 2, 3, 4, 5, 6, 7, 0, 8)) == (5, 7, 8)
    assert adjacent_tiles(GOAL) == (6, 8)
    assert adjacent_tiles((0, 1, 2, 3, 4, 5, 6, 7, 8)) == (1, 3)


def test_slide_tile_swaps_an_adjacent_numbered_tile() -> None:
    board = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    assert slide_tile(board, 8) == GOAL
    assert slide_tile(GOAL, 8) == board


@pytest.mark.parametrize("tile", [0, 9, "8", 8.0, True, None])
def test_slide_tile_rejects_invalid_tile_type_or_range(tile: object) -> None:
    with pytest.raises(ValueError, match="integer from 1 through 8"):
        slide_tile(GOAL, tile)  # type: ignore[arg-type]


def test_slide_tile_rejects_non_adjacent_tile() -> None:
    board = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    with pytest.raises(ValueError, match="not adjacent"):
        slide_tile(board, 1)


def test_solver_returns_tile_ids_that_replay_to_goal() -> None:
    board = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    actions = solve(board)
    assert actions == [8]
    assert all(type(action) is int and 1 <= action <= 8 for action in actions)

    replayed = board
    for action in actions:
        assert action in adjacent_tiles(replayed)
        replayed = slide_tile(replayed, action)
    assert replayed == GOAL


def test_exact_distance_table_has_expected_reachable_states() -> None:
    hardest = (8, 6, 7, 2, 5, 4, 3, 0, 1)
    assert exact_distance(hardest) == 31

    import puzzle3.solver as solver_module

    assert solver_module._DISTANCE_TABLE is not None
    assert len(solver_module._DISTANCE_TABLE) == 181_440
    assert max(solver_module._DISTANCE_TABLE.values()) == 31


def test_regenerated_eval_files_have_valid_tile_action_records(tmp_path) -> None:
    import runpy

    generator = runpy.run_path("data/generate_eval_data_3x3.py")
    records = generator["generate_eval_candidates"](45, random.Random(42))
    jsonl_path = tmp_path / "eval.jsonl"
    parquet_path = tmp_path / "eval.parquet"
    generator["write_eval_jsonl"](records, jsonl_path)
    generator["write_eval_parquet"](records, parquet_path)

    for path in (jsonl_path, parquet_path):
        examples = load_examples(dataset=str(path))
        assert len(examples) == 45
        assert Counter(example.metadata["bucket"] for example in examples) == {
            "easy": 15,
            "medium": 15,
            "hard": 15,
        }
        for example in examples:
            assert example.metadata["action_interface"] == "tile_id_v1"
            assert all(type(action) is int for action in example.optimal_actions)
            assert len(example.optimal_actions) == example.optimal_length


def test_exhaustive_eval_has_requested_unseen_boards() -> None:
    import runpy

    generator = runpy.run_path("data/create_exhaustive_eval.py")
    depth_counts = generator["DEPTH_COUNTS"]
    generate = generator["generate_exhaustive_eval_candidates"]
    excluded = {
        (1, 2, 3, 4, 5, 6, 7, 0, 8),
        (1, 2, 3, 4, 5, 6, 0, 7, 8),
    }
    records = generate(random.Random(42), excluded)

    assert len(records) == 272
    assert Counter(record["optimal_length"] for record in records) == depth_counts
    assert not ({tuple(record["board"]) for record in records} & excluded)


def test_sft_source_is_balanced_fresh_and_has_no_oracle_actions() -> None:
    import runpy

    generator = runpy.run_path("data/create_sft_source_3x3.py")
    paths = generator["enumerate_from_goal"]()
    excluded = {next(board for board, path in paths.items() if len(path) == 12)}
    records = generator["generate_sft_source"](random.Random(42), excluded)

    assert len(records) == 1_500
    assert Counter(record["optimal_length"] for record in records) == {
        depth: 300 for depth in range(12, 17)
    }
    assert len({tuple(record["board"]) for record in records}) == len(records)
    assert not ({tuple(record["board"]) for record in records} & excluded)
    assert all(record["action_interface"] == "tile_id_v1" for record in records)
    assert all("optimal_actions" not in record for record in records)




def test_scramble_never_immediately_reverses_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_slide_tile = scramble_module.slide_tile
    selected: list[int] = []

    def recording_slide(board: tuple[int, ...], tile: int) -> tuple[int, ...]:
        if selected:
            assert tile != selected[-1]
        selected.append(tile)
        return real_slide_tile(board, tile)

    monkeypatch.setattr(scramble_module, "slide_tile", recording_slide)
    scrambled = scramble_module.scramble_board(30, random.Random(42))

    assert len(selected) == 30
    assert scrambled != GOAL
