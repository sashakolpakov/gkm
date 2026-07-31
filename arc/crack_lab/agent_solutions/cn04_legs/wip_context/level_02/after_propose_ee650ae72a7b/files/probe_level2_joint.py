"""Check whether completing an A-to-B socket makes later B motion carry A."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1
from probe_level2_collect import covered, peg_centers
from probe_level2_relay import A_ROUTES


def bodies(node):
    return [
        (blob.color, blob.area, blob.bbox)
        for blob in perception.connected_components(
            node.frame(), colors=(0, 9, 11, 14, 15), min_area=4
        )
    ]


def select_color(node, color):
    ys, xs = np.where(perception.arr(node.frame()) == color)
    node.step(6, int(xs[0]), int(ys[0]))


def probe(env):
    play_level_1(env)
    pegs = peg_centers(env.frame())
    for pair, route in A_ROUTES.items():
        for extra_turns in range(4):
            node = perception.replay(env, route + [5] * extra_turns)
            hit = tuple(i for i, peg in enumerate(pegs) if covered(node, peg))
            select_color(node, 14)
            selected = bodies(node)
            before = perception.arr(node.frame()).copy()
            node.step(2)
            delta = perception.frame_delta(before, node.frame())
            print(
                "trial", pair, extra_turns, "hit", hit,
                "selected", selected, "after", bodies(node),
                "move_delta", (delta["count"], delta["bbox"]),
            )


arena.run_program("cn04", probe)
