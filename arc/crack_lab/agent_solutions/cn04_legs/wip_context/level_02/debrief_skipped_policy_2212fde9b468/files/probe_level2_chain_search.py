"""Search reusable mover segments: engage source handles, then bridge target."""
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1
from probe_level2_collect import avatar_key, covered, peg_centers


GROUPS = {"A": (1, 3), "B": (0, 2, 4, 5), "C": (6, 7, 8, 11),
          "D": (9, 10)}


def bridge_axis(node, peg, target_color):
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


def select_color(node, color):
    ys, xs = np.where(perception.arr(node.frame()) == color)
    node.step(6, int(xs[0]), int(ys[0]))


def search(root, pegs, source_group, target_group, target_color,
           max_states=16000, max_depth=70):
    start_mask = sum(
        1 << i for i in source_group if covered(root, pegs[i])
    )
    wanted_mask = sum(1 << i for i in source_group)
    q = deque([(root.clone(), [], start_mask)])
    seen = {(avatar_key(root), start_mask)}
    while q and len(seen) < max_states:
        node, path, mask = q.popleft()
        bridges = tuple(
            i for i in target_group
            if bridge_axis(node, pegs[i], target_color)
        )
        if mask == wanted_mask and len(bridges) >= 2:
            return path, bridges, len(seen)
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4, 5):
            child = node.clone()
            child.step(action)
            child_mask = mask | sum(
                1 << i
                for i in source_group
                if covered(child, pegs[i])
            )
            key = (avatar_key(child), child_mask)
            if key not in seen:
                seen.add(key)
                q.append((child, path + [action], child_mask))
    return None, (), len(seen)


def probe(env):
    play_level_1(env)
    pegs = peg_centers(env.frame())
    trials = (
        ("A", None, GROUPS["A"], GROUPS["B"], 14),
        ("B", 14, GROUPS["B"], GROUPS["C"], 11),
        ("C", 11, GROUPS["C"], GROUPS["D"], 9),
    )
    for name, color, source, target, target_color in trials:
        node = env.clone()
        if color is not None:
            select_color(node, color)
        path, bridges, states = search(
            node, pegs, source, target, target_color
        )
        print("agent", name, "states", states, "bridges", bridges,
              "path", path)


arena.run_program("cn04", probe)
