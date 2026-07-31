"""Replay the collision-free A-C-B-D handle assembly."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1


def select_color(node, color):
    ys, xs = np.where(perception.arr(node.frame()) == color)
    action = (6, int(xs[0]), int(ys[0]))
    node.step(*action)
    return action


def black_cells(node):
    a = perception.arr(node.frame())
    return {
        (r, c)
        for r in range(3, 61, 3)
        for c in range(3, 61, 3)
        if int(a[r, c]) == 0
    }


def apply(node, path, actions):
    for action in actions:
        node.step(action)
        path.append(action)
        if node.levels_completed > 1:
            return True
    return False


def probe(env):
    play_level_1(env)
    node = env.clone()
    path = []
    apply(node, path, [1] * 2 + [4] * 4)
    path.append(select_color(node, 11))
    apply(node, path, [1] * 8 + [4] * 4)
    path.append(select_color(node, 9))
    apply(node, path, [5] * 3)
    body = black_cells(node)
    top_left = (min(r for r, _ in body), min(c for _, c in body))
    print("D_rotated_top_left", top_left)
    vertical = [1] * ((top_left[0] - 21) // 3)
    horizontal = [4] * ((54 - top_left[1]) // 3)
    solved = apply(node, path, vertical + horizontal)
    print(
        "result",
        node.levels_completed,
        "gray",
        perception.color_counts(node.frame()).get(8, 0),
        "solved",
        solved,
        "path",
        path,
    )


arena.run_program("cn04", probe)
