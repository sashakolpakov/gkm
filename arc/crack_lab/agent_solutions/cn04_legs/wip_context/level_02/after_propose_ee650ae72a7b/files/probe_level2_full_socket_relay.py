"""Test full-socket visit routes followed by visible bridge poses."""
import sys
from itertools import product

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1


ROUTES = {
    "A": [2] + [4] * 14,
    "B": [5, 5, 2, 2, 2, 2, 2, 3, 2, 2] + [3] * 9,
    "C": [2, 1, 3, 5] + [4] * 9 + [1],
}
COLORS = {"B": 14, "C": 11, "D": 9}


def select_color(node, color):
    ys, xs = np.where(perception.arr(node.frame()) == color)
    action = (6, int(xs[0]), int(ys[0]))
    node.step(*action)
    return action


def probe(env):
    play_level_1(env)
    root = env.clone()
    for turns in product(range(4), repeat=4):
        node = root.clone()
        path = []
        for index, name in enumerate("ABC"):
            if name != "A":
                path.append(select_color(node, COLORS[name]))
            for action in ROUTES[name] + [5] * turns[index]:
                node.step(action)
                path.append(action)
                if node.levels_completed > 1:
                    print("solved", turns, len(path), path)
                    return
        path.append(select_color(node, COLORS["D"]))
        for _ in range(turns[3]):
            node.step(5)
            path.append(5)
            if node.levels_completed > 1:
                print("solved", turns, len(path), path)
                return
    print("unsolved", 4 ** 4)


arena.run_program("cn04", probe)
