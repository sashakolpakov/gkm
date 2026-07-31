"""Find routes that visit both socket ends before settling into a bridge."""
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1
from probe_level2_collect import avatar_key, covered, peg_centers


SOURCE = {"A": (1, 3), "B": (0, 2, 4, 5), "C": (6, 7, 8, 11)}
TARGET_PAIRS = {"A": ((0, 5), (2, 4)), "B": ((6, 8),),
                "C": ((9, 10),)}
TARGET_COLOR = {"A": 14, "B": 11, "C": 9}
SELECT_COLOR = {"B": 14, "C": 11}


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


def search(root, pegs, required, target_pair, target_color,
           max_states=30000, max_depth=80):
    bits = {peg: 1 << offset for offset, peg in enumerate(required)}
    wanted = (1 << len(required)) - 1

    def advance(node, mask):
        for peg, bit in bits.items():
            if covered(node, pegs[peg]):
                mask |= bit
        return mask

    start_mask = advance(root, 0)
    q = deque([(root.clone(), [], start_mask)])
    seen = {(avatar_key(root), start_mask)}
    while q and len(seen) < max_states:
        node, path, mask = q.popleft()
        if mask == wanted and all(
            bridged(node, pegs[i], target_color) for i in target_pair
        ):
            return path, len(seen)
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4, 5):
            child = node.clone()
            child.step(action)
            child_mask = advance(child, mask)
            key = (avatar_key(child), child_mask)
            if key not in seen:
                seen.add(key)
                q.append((child, path + [action], child_mask))
    return None, len(seen)


def probe(env):
    play_level_1(env)
    pegs = peg_centers(env.frame())
    for name in "ABC":
        for pair in TARGET_PAIRS[name]:
            node = env.clone()
            if name in SELECT_COLOR:
                select_color(node, SELECT_COLOR[name])
            required = tuple(dict.fromkeys(SOURCE[name] + pair))
            path, states = search(
                node, pegs, required, pair, TARGET_COLOR[name]
            )
            print("agent", name, "pair", pair, "states", states,
                  "path", path)


arena.run_program("cn04", probe)
