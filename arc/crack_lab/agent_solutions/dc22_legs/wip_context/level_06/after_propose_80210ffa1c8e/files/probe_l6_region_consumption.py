"""Measure exact movement-state and consumable-floor progress in each region."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import MAIN, enter_right


REGIONS = {
    "right0": (slice(48, 54), slice(32, 38)),
    "right3": (slice(56, 62), slice(32, 38)),
    "top": (slice(4, 12), slice(4, 12)),
    "hub": (slice(48, 54), slice(18, 30)),
}


def frame_key(env):
    return perception.arr(env.frame())[:63].tobytes()


def remaining(env, region):
    rows, cols = REGIONS[region]
    block = perception.arr(env.frame())[rows, cols].copy()
    block[block == 14] = 2
    return int(((block != 2) & (block != 4)).sum()), block.tolist()


def exact_closure(root, region, max_states=700):
    base_level = root.levels_completed
    queue = deque([(root.clone(), [])])
    seen = {frame_key(root)}
    best = (remaining(root, region)[0], [], remaining(root, region)[1])
    while queue and len(seen) < max_states:
        node, path = queue.popleft()
        score, block = remaining(node, region)
        if score < best[0]:
            best = score, path, block
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_path = path + [action]
            if child.levels_completed > base_level:
                return len(seen), best, child_path
            key = frame_key(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, child_path))
    return len(seen), best, None


def observe(env):
    solve.solve(env)
    roots = {
        "right0": enter_right(env, 0),
        "right3": enter_right(env, 3),
        "top": enter_right(env, 2),
    }
    hub = roots["right3"].clone()
    hub.step(*MAIN)
    roots["hub"] = hub
    for name, root in roots.items():
        states, best, win = exact_closure(root, name)
        print(
            "REGION_CONSUMPTION", name, "STATES", states,
            "BEST", best[0], "PATH", best[1], "WIN", win,
            "BLOCK", best[2],
        )


arena.run_program("dc22", observe)
