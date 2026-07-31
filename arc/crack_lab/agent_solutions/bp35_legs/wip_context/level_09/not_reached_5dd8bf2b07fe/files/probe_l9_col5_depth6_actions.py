"""Probe the last safe row of direct lane five before its lethal descent."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


def objects(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 9, 11, 12, 14), min_area=2
        )
        if blob.bbox[0] < 63
    )


def col5_depth6(root):
    child = replay(root, route(), skips=SKIPS)
    child.step(*controls(child)[0])
    for _ in range(5):
        child.step(4)
    for _ in range(6):
        child.step(6, 33, 35)
    return child


def probe(env):
    enter_level_9(env)
    root = col5_depth6(env)
    print("FRONTIER", compact(root), "objects", objects(root))
    actions = [3, 4, 7]
    actions.extend(
        (6, x, y)
        for y in (17, 21, 27, 33)
        for x in (21, 27, 33, 39, 45)
    )
    for action in actions:
        child = root.clone()
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print(
            "ACTION",
            action,
            "terminal",
            bool(child.terminal()),
            "level",
            int(child.levels_completed) + 1,
            "state",
            compact(child),
            "objects",
            objects(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
