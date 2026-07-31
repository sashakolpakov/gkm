"""Greedily align two movable handles at each chain stage."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1
from probe_level2_collect import avatar_key


def gray_count(node):
    return perception.color_counts(node.frame()).get(8, 0)


def select_color(node, color):
    ys, xs = np.where(perception.arr(node.frame()) == color)
    action = (6, int(xs[0]), int(ys[0]))
    node.step(*action)
    return action


def probe(env):
    play_level_1(env)
    node = env.clone()
    total = []
    for name, color, wanted in (
        ("A", None, 90),
        ("C", 11, 72),
        ("D", 9, 54),
    ):
        if color is not None:
            total.append(select_color(node, color))
        segment = perception.bounded_bfs(
            node,
            lambda child, _: child.levels_completed > 1
            or gray_count(child) <= wanted,
            actions=(1, 2, 3, 4, 5),
            key_fn=avatar_key,
            max_states=4000,
            max_depth=45,
        )
        print(name, "before", gray_count(node), "segment", segment)
        if segment is None:
            return
        for action in segment:
            node.step(action)
            total.append(action)
        print(
            name,
            "after",
            gray_count(node),
            "level",
            node.levels_completed,
            "gray",
            [
                (b.area, b.bbox)
                for b in perception.connected_components(
                    node.frame(), colors=(8,), min_area=1
                )
            ],
        )
        if node.levels_completed > 1:
            print("solved", len(total), total)
            return
    print("final", node.levels_completed, len(total), total)


arena.run_program("cn04", probe)
