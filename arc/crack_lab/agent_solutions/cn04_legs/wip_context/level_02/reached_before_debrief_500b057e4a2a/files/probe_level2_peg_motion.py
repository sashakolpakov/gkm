"""Check whether gray peg components move with selected bodies."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1


def centers(frame):
    return tuple(
        (int(round(b.centroid[0])), int(round(b.centroid[1])))
        for b in perception.connected_components(frame, colors=(8,), min_area=4)
    )


def probe(env):
    play_level_1(env)
    original = centers(env.frame())
    for action in (1, 4, 5):
        child = env.clone()
        child.step(action)
        now = centers(child.frame())
        print(
            0,
            action,
            "gone",
            tuple(point for point in original if point not in now),
            "new",
            tuple(point for point in now if point not in original),
        )
    for color in (14, 11, 9):
        node = env.clone()
        ys, xs = np.where(perception.arr(node.frame()) == color)
        node.step(6, int(xs[0]), int(ys[0]))
        for action in (1, 4, 5):
            child = node.clone()
            child.step(action)
            now = centers(child.frame())
            print(
                color,
                action,
                "gone",
                tuple(point for point in original if point not in now),
                "new",
                tuple(point for point in now if point not in original),
            )


arena.run_program("cn04", probe)
