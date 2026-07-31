"""Build the full second-chamber bridge on the compressed prefix, then flip."""

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


def bridged_flip(root):
    child = replay(root, route(), skips=SKIPS)
    for x in (57, 51, 45, 39, 33, 27, 21, 15, 9):
        child.step(6, x, 45)
    child.step(*controls(child)[0])
    return child


def trace(root, name, action):
    child = bridged_flip(root)
    print(name, "START", compact(child), "objects", objects(child))
    for count in range(1, 13):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print(
            name,
            count,
            "terminal",
            bool(child.terminal()),
            "level",
            int(child.levels_completed) + 1,
            "controls",
            controls(child),
            "state",
            compact(child),
            "objects",
            objects(child),
        )
        if child.terminal() or int(child.levels_completed) >= 9:
            return


def probe(env):
    enter_level_9(env)
    trace(env, "RIGHT", 4)
    trace(env, "DOWN", (6, 3, 35))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
