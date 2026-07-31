"""Find and replay reverse-chain bridge poses D-to-C-to-B-to-A."""
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1
from probe_level2_collect import avatar_key, peg_centers


GROUPS = {"A": (1, 3), "B": (0, 2, 4, 5), "C": (6, 7, 8, 11)}


def select_color(node, color):
    ys, xs = np.where(perception.arr(node.frame()) == color)
    action = (6, int(xs[0]), int(ys[0]))
    node.step(*action)
    return action


def bridge(node, peg, target_color):
    a = perception.arr(node.frame())
    r, c = peg
    if int(a[r, c]) != 8:
        return False
    return any(
        {first, second} == {0, target_color}
        for first, second in (
            (int(a[r - 3, c]), int(a[r + 3, c])),
            (int(a[r, c - 3]), int(a[r, c + 3])),
        )
    )


def search(root, pegs, target_group, target_color):
    q = deque([(root.clone(), [])])
    seen = {avatar_key(root)}
    while q and len(seen) < 4000:
        node, path = q.popleft()
        hits = tuple(i for i in target_group
                     if bridge(node, pegs[i], target_color))
        if len(hits) >= 2:
            return path, hits, len(seen)
        if len(path) >= 50:
            continue
        for action in (1, 2, 3, 4, 5):
            child = node.clone()
            child.step(action)
            key = avatar_key(child)
            if key not in seen:
                seen.add(key)
                q.append((child, path + [action]))
    return None, (), len(seen)


def probe(env):
    play_level_1(env)
    root = env.clone()
    pegs = peg_centers(root.frame())
    trials = (
        ("D", 9, GROUPS["C"], 11),
        ("C", 11, GROUPS["B"], 14),
        ("B", 14, GROUPS["A"], 15),
    )
    routes = {}
    for name, color, target_group, target_color in trials:
        node = root.clone()
        select_color(node, color)
        route, hits, states = search(node, pegs, target_group, target_color)
        routes[name] = route
        print("agent", name, "states", states, "hits", hits, "path", route)
    if any(route is None for route in routes.values()):
        return
    node = root.clone()
    full_path = []
    for name, color, _, _ in trials:
        full_path.append(select_color(node, color))
        for action in routes[name]:
            node.step(action)
            full_path.append(action)
            if node.levels_completed > 1:
                print("solved", name, full_path)
                return
    print("unsolved", full_path)


arena.run_program("cn04", probe)
