from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from sft_generation.storage import write_jsonl


_SPEC = importlib.util.spec_from_file_location(
    "build_sft_dataset",
    Path(__file__).parents[1] / "scripts" / "build_sft_dataset.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
puzzle_validation_record = _MODULE.puzzle_validation_record


def test_sft_jsonl_preserves_prompt_first_column_order(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(
        path,
        [
            {
                "prompt": [],
                "completion": [],
                "metadata": {},
                "tools": [],
            }
        ],
        sort_keys=False,
    )

    assert list(json.loads(path.read_text(encoding="utf-8"))) == [
        "prompt",
        "completion",
        "metadata",
        "tools",
    ]


def test_puzzle_validation_record_has_solution_metadata_but_no_training_target() -> None:
    record = puzzle_validation_record(
        {
            "id": "sft-validation-0001",
            "board": [1, 2, 3, 4, 5, 6, 0, 7, 8],
            "optimal_actions": [7, 8],
            "optimal_length": 2,
            "action_interface": "tile_id_v1",
        }
    )

    assert record["prompt"] == []
    assert record["completion"] == []
    assert record["tools"] == []
    assert record["metadata"]["record_type"] == "puzzle"
    assert record["metadata"]["initial_depth"] == 2
    assert record["metadata"]["optimal_actions"] == [7, 8]
    assert record["metadata"]["legal_tiles"] == [4, 7]
