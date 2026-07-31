"""Test the A-to-B-to-C-to-D cooperative relay."""
import sys
from itertools import product

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

from players import play_level_1


A_ROUTES = {
    (0, 2): [2] + [4] * 8 + [5],
    (0, 5): [2] + [4] * 9,
    (2, 4): [2] * 3 + [4] * 8 + [5],
}
B_ROUTES = {
    (6, 7): [2] * 5 + [3] * 7 + [5],
    (6, 8): [2] * 4 + [3] * 10,
    (7, 11): [2] * 7 + [3] * 5,
    (8, 11): [2] * 9 + [3] * 8 + [5, 2],
}
C_ROUTE = [4] * 9


def select_color(node, color):
    ys, xs = np.where(np.asarray(node.frame()) == color)
    node.step(6, int(xs[0]), int(ys[0]))
    return (6, int(xs[0]), int(ys[0]))


def apply(node, path, actions):
    for action in actions:
        node.step(action)
        path.append(action)
        if node.levels_completed > 1:
            return True
    return False


def probe(env):
    play_level_1(env)
    root = env.clone()
    trials = 0
    for a_item, b_item, turns in product(
        A_ROUTES.items(), B_ROUTES.items(), product(range(4), repeat=3)
    ):
        (a_pair, a_route), (b_pair, b_route), (ta, tb, tc) = (
            a_item, b_item, turns
        )
        trials += 1
        node = root.clone()
        path = []
        if apply(node, path, a_route + [5] * ta):
            print("solved", trials, a_pair, b_pair, turns, path)
            return
        path.append(select_color(node, 14))
        if node.levels_completed > 1 or apply(node, path, b_route + [5] * tb):
            print("solved", trials, a_pair, b_pair, turns, path)
            return
        path.append(select_color(node, 11))
        if node.levels_completed > 1 or apply(node, path, C_ROUTE + [5] * tc):
            print("solved", trials, a_pair, b_pair, turns, path)
            return
    print("unsolved", trials)


arena.run_program("cn04", probe)
