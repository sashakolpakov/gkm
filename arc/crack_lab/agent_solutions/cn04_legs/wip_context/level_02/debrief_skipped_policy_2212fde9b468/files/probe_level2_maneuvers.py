"""Find level-2 relay segments that reproduce level 1's three-turn joint."""
import sys
from collections import deque
from itertools import combinations

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1
from probe_level2_collect import avatar_key, covered, peg_centers


GROUPS = {"A": (1, 3), "B": (0, 2, 4, 5), "C": (6, 7, 8, 11),
          "D": (9, 10)}


def select_color(node, color):
    ys, xs = np.where(perception.arr(node.frame()) == color)
    node.step(6, int(xs[0]), int(ys[0]))


def bridged(node, peg, target_color):
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


def search(root, pegs, source_group, target_group, target_color,
           max_states=18000, max_depth=65):
    wanted = sum(1 << i for i in source_group)
    start_mask = sum(1 << i for i in source_group if covered(root, pegs[i]))
    q = deque([(root.clone(), [], start_mask)])
    seen = {(avatar_key(root), start_mask)}
    while q and len(seen) < max_states:
        node, path, mask = q.popleft()
        if mask == wanted:
            turn1 = node.clone()
            turn1.step(5)
            hits = tuple(i for i in target_group if covered(turn1, pegs[i]))
            if len(hits) >= 2:
                turn3 = turn1.clone()
                turn3.step(5)
                turn3.step(5)
                for pair in combinations(hits, 2):
                    if all(bridged(turn3, pegs[i], target_color) for i in pair):
                        return path + [5, 5, 5], pair, len(seen)
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4, 5):
            child = node.clone()
            child.step(action)
            child_mask = mask | sum(
                1 << i for i in source_group if covered(child, pegs[i])
            )
            key = (avatar_key(child), child_mask)
            if key not in seen:
                seen.add(key)
                q.append((child, path + [action], child_mask))
    return None, (), len(seen)


def probe(env):
    play_level_1(env)
    pegs = peg_centers(env.frame())
    for name, color, source, target, target_color in (
        ("A", None, GROUPS["A"], GROUPS["B"], 14),
        ("B", 14, GROUPS["B"], GROUPS["C"], 11),
        ("C", 11, GROUPS["C"], GROUPS["D"], 9),
    ):
        node = env.clone()
        if color is not None:
            select_color(node, color)
        path, pair, states = search(
            node, pegs, source, target, target_color
        )
        print("agent", name, "states", states, "pair", pair, "path", path)


arena.run_program("cn04", probe)
