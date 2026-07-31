"""Test whether a cargo terminal unlocks the horizontal left exit."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve
from probe_l6_cargo_top import CARGO_TOP_PATH
from probe_l6_ring_route import HUB_TO_BRIDGE
from probe_l6_right import MAIN, TOP, enter_right


UP = (6, 50, 34)
DOWN = (6, 50, 40)
LEFT = (6, 46, 36)
RIGHT = (6, 54, 36)


CASES = {
    "TOP": (CARGO_TOP_PATH, 2),
    "FAR_LEFT": ([3, LEFT, LEFT, LEFT, LEFT], 4),
    "LOWER_LEFT": ([2, DOWN, 1, 3, LEFT], 4),
    "LOWER_RIGHT": ([2, DOWN, 1, 4, RIGHT], 3),
    "RIGHT": ([4, RIGHT], 3),
}


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    for label, (cargo_path, return_to_center) in CASES.items():
        node = enter_right(env, 3)
        for action in cargo_path:
            if isinstance(action, tuple):
                node.step(*action)
            else:
                node.step(action)
        node.step(return_to_center)
        node.step(*MAIN)
        for action in HUB_TO_BRIDGE:
            node.step(action)
        node.step(*TOP)
        node.step(1)
        node.step(3)
        node.step(*TOP)
        for _ in range(12):
            node.step(1)
        node.step(*MAIN)
        for _ in range(4):
            node.step(3)
        node.step(3)
        print(
            "EXIT_BY_CARGO", label, node.levels_completed,
            node.levels_completed - base_level,
        )


arena.run_program("dc22", observe)
