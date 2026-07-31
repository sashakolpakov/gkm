"""Try every order of the independently verified forward bridge poses."""
import sys
from itertools import permutations, product

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1


VARIANTS = {
    "A": (
        [4] * 8 + [5, 5],
        [4] * 14,
        [2] + [4] * 14,
        [2] * 2 + [4] * 16,
        [4] * 11 + [5, 5] + [4] * 3,
        [2] * 3 + [4] * 14,
    ),
    "B": (
        [2] * 7 + [3] * 10 + [5, 5],
    ),
    "C": (
        [1] + [4] * 9,
        [1] + [4] * 8 + [5],
    ),
}
COLORS = {"A": 15, "B": 14, "C": 11}


def select(node, name):
    ys, xs = np.where(perception.arr(node.frame()) == COLORS[name])
    action = (6, int(xs[0]), int(ys[0]))
    node.step(*action)
    return action


def probe(env):
    play_level_1(env)
    root = env.clone()
    trials = 0
    for a_route, b_route, c_route in product(
        VARIANTS["A"], VARIANTS["B"], VARIANTS["C"]
    ):
        routes = {"A": a_route, "B": b_route, "C": c_route}
        for order in permutations("ABC"):
            trials += 1
            node = root.clone()
            active = "A"
            path = []
            for name in order:
                if active != name:
                    path.append(select(node, name))
                    active = name
                for action in routes[name]:
                    node.step(action)
                    path.append(action)
                    if node.levels_completed > 1:
                        print("solved", "".join(order), path)
                        return
    print("unsolved", "trials", trials)


arena.run_program("cn04", probe)
