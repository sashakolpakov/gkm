"""Test each cardinal cargo terminal with the matching avatar exit."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve
from probe_l6_cargo_top import CARGO_TOP_PATH
from probe_l6_right import enter_right


UP = (6, 50, 34)
DOWN = (6, 50, 40)
LEFT = (6, 46, 36)
RIGHT = (6, 54, 36)
CASES = {
    "UP": (CARGO_TOP_PATH, 1),
    "DOWN": ([2, DOWN], 2),
    "LEFT": ([3, LEFT, LEFT, LEFT, LEFT], 3),
    "RIGHT": ([4, RIGHT], 4),
}


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    for label, (path, outward) in CASES.items():
        node = enter_right(env, 3)
        for action in path:
            if isinstance(action, tuple):
                node.step(*action)
            else:
                node.step(action)
        node.step(outward)
        print(
            "CARDINAL_EXIT", label, node.levels_completed,
            node.levels_completed - base_level, node.terminal(),
        )


arena.run_program("dc22", observe)
