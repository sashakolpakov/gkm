"""Verify the exact handle geometry of the rewarded level-1 pose."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception


def centers(frame):
    return tuple(
        (int(round(b.centroid[0])), int(round(b.centroid[1])))
        for b in perception.connected_components(frame, colors=(8,), min_area=4)
    )


def select_color(node, color):
    ys, xs = np.where(perception.arr(node.frame()) == color)
    node.step(6, int(xs[0]), int(ys[0]))


def probe(env):
    print("initial", centers(env.frame()))
    node = env.clone()
    select_color(node, 14)
    node.step(4)
    select_color(node, 12)
    for action in [2] * 7 + [4] * 4:
        node.step(action)
    print("prefinish_with_B_shifted", centers(node.frame()))
    for turn in range(1, 5):
        node.step(5)
        print("turn", turn, "level", node.levels_completed, "gray", centers(node.frame()))


arena.run_program("cn04", probe)
