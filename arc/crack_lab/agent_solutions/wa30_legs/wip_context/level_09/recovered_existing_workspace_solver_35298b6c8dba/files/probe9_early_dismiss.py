"""Beam-search a compact level-9 interception before remote deliveries."""

import gkm_try

from perception import arr
from probe9_prefix_shortcuts import reach_level_9
from probe9_verify import boxes


def distance(env):
    avatar = boxes(env.frame(), 14)
    thief = boxes(env.frame(), 15)
    if not thief:
        return -10000
    if not avatar:
        return 10000
    ar, ac, _, _ = avatar[0]
    tr, tc, _, _ = thief[0]
    return abs(ar - tr) + abs(ac - tc)


def search(start, width=300, max_depth=28):
    frontier = [(start.clone(), [])]
    transitions = 0
    for depth in range(1, max_depth + 1):
        candidates = {}
        for node, path in frontier:
            for action in (5, 1, 2, 3, 4):
                child = node.clone()
                child.step(action)
                transitions += 1
                child_path = path + [action]
                if not boxes(child.frame(), 15):
                    return child_path, transitions
                key = arr(child.frame()).tobytes()
                candidates.setdefault(key, (child, child_path))
        ranked = sorted(
            candidates.values(),
            key=lambda item: (distance(item[0]), item[1].count(5)),
        )
        frontier = ranked[:width]
        print(
            "EARLY_DISMISS_DEPTH",
            depth,
            transitions,
            distance(frontier[0][0]),
            frontier[0][1],
            flush=True,
        )
    return None, transitions


def inspect(env):
    reach_level_9(env)
    path, transitions = search(env)
    print("EARLY_DISMISS_RESULT", path, transitions, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
