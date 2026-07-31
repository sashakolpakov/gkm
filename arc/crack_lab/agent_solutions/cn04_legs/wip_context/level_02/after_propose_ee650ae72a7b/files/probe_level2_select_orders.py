"""Test whether selection order itself advances level 2."""
import sys
from itertools import permutations

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1


POINTS = {"A": (12, 12), "B": (45, 15), "C": (18, 39), "D": (51, 51)}


def probe(env):
    play_level_1(env)
    for order in permutations("BCD"):
        node = env.clone()
        for name in order + (order[-1],):
            node.step(6, *POINTS[name])
            if node.levels_completed > 1:
                print("solved", order, "at", name)
                return
        print(
            "order", "".join(order), "level", node.levels_completed,
            "colors", perception.color_counts(node.frame()),
        )


arena.run_program("cn04", probe)
