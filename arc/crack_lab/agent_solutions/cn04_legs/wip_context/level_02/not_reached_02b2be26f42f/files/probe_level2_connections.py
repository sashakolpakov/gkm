"""Find poses that bridge two fixed pegs between an active and target body."""
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1
from probe_level2_collect import avatar_key, peg_centers


GROUPS = {"A": (1, 3), "B": (0, 2, 4, 5), "C": (6, 7, 8, 11),
          "D": (9, 10)}
SELECT = {"B": 14, "C": 11}


def select_color(node, color):
    ys, xs = np.where(perception.arr(node.frame()) == color)
    node.step(6, int(xs[0]), int(ys[0]))


def bridge_axis(node, peg, target_color):
    a = perception.arr(node.frame())
    r, c = peg
    if int(a[r, c]) != 8:
        return None
    sides = (
        ("v", int(a[r - 3, c]), int(a[r + 3, c])),
        ("h", int(a[r, c - 3]), int(a[r, c + 3])),
    )
    for axis, first, second in sides:
        if {first, second} == {0, target_color}:
            return axis
    return None


def find_bridges(root, pegs, target_group, target_color):
    q = deque([(root.clone(), [])])
    seen = {avatar_key(root)}
    found = {}
    while q and len(seen) < 3000:
        node, path = q.popleft()
        hits = tuple(
            (i, bridge_axis(node, pegs[i], target_color))
            for i in target_group
            if bridge_axis(node, pegs[i], target_color)
        )
        if len(hits) >= 2:
            signature = tuple(hits)
            found.setdefault(signature, path)
        if len(path) >= 45:
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
    a = perception.arr(env.frame())
    for index, (r, c) in enumerate(pegs):
        neighbors = tuple(int(a[r + dr, c + dc])
                          for dr, dc in ((-3, 0), (3, 0), (0, -3), (0, 3)))
        print("peg", index, (r, c), "neighbors", neighbors)

    agents = [("A", env.clone(), GROUPS["B"], 14)]
    for name, color, target, target_color in (
        ("B", 14, GROUPS["C"], 11),
        ("C", 11, GROUPS["D"], 9),
    ):
        node = env.clone()
        select_color(node, color)
        agents.append((name, node, target, target_color))

    for name, node, target, target_color in agents:
        states, found = find_bridges(node, pegs, target, target_color)
        print("agent", name, "states", states,
              "bridges", sorted(found.items()))


arena.run_program("cn04", probe)
