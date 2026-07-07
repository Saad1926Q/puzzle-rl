from puzzle.board import GOAL
from puzzle.render import render


def test_render_goal():
    expected = "1 2 3 4\n5 6 7 8\n9 10 11 12\n13 14 15 _"
    assert render(GOAL) == expected
