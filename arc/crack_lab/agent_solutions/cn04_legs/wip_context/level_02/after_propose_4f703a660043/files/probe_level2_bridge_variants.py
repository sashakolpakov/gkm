"""Enumerate every exact active-to-target bridge pose for each forward edge."""
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1
from probe_level2_collect import avatar_key, peg_centers


GROUPS = {"B": (0, 2, 4, 5), "C": (6, 7, 8, 11), "D": (9, 10)}


def select_color(node, color):
    ys, xs = np.where(perception.arr(node.frame()) == color)
    node.step(6, int(xs[0]), int(ys[0]))


def bridge(node, peg, target_color):
    a = perception.arr(node.frame())
    r, c = peg
    if int(a[r, c]) != 8:
        return None
    for axis, first, second in (
        ("v", int(a[r - 3, c]), int(a[r + 3, c])),
        ("h", int(a[r, c - 3]), int(a[r, c + 3])),
    ):
        if {first, second} == {0, target_color}:
            return axis
    return None


def variants(root, pegs, target_group, target_color):
    q = deque([(root.clone(), [])])
    seen = {avatar_key(root)}
    found = []
    while q and len(seen) < 4000:
        node, path = q.popleft()
        hits = tuple(
            (i, bridge(node, pegs[i], target_color))
            for i in target_group
            if bridge(node, pegs[i], target_color)
        )
        if len(hits) >= 2:
            blobs = perception.connected_components(
                node.frame(), colors=(0,), min_area=4
            )
            body = max((b for b in blobs if b.bbox[0] > 0),
                       key=lambda b: b.area)
            found.append((hits, body.bbox, path))
        if len(path) >= 50:
            continue
        for action in (1, 2, 3, 4, 5):
            child = node.clone()
            child.step(action)
            key = avatar_key(child)
            if key not in seen:
                seen.add(key)
                q.append((child, path + [action]))
    return len(seen), found


def probe(env):
    play_level_1(env)
    pegs = peg_centers(env.frame())
    for name, color, target, target_color in (
        ("A", None, GROUPS["B"], 14),
        ("B", 14, GROUPS["C"], 11),
        ("C", 11, GROUPS["D"], 9),
    ):
        node = env.clone()
        if color is not None:
            select_color(node, color)
        states, found = variants(node, pegs, target, target_color)
        print("agent", name, "states", states, "variants", len(found))
        for index, item in enumerate(found):
            print("variant", name, index, item)


arena.run_program("cn04", probe)
