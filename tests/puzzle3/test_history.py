from __future__ import annotations

import pytest

from puzzle3.history import LastTurns


def test_last_turns_keeps_latest_complete_turns_in_order() -> None:
    policy = LastTurns(max_turns=4)

    assert policy.select([0, 1, 2, 3, 4, 5]) == (2, 3, 4, 5)


def test_last_turns_does_not_mutate_source_history() -> None:
    history = [0, 1, 2]

    selected = LastTurns(max_turns=2).select(history)
    history.append(3)

    assert selected == (1, 2)
    assert history == [0, 1, 2, 3]


def test_zero_history_is_supported() -> None:
    assert LastTurns(max_turns=0).select([1, 2]) == ()


@pytest.mark.parametrize("max_turns", [-1, -4])
def test_negative_history_window_is_rejected(max_turns: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LastTurns(max_turns=max_turns)
