"""Trace the two exits from chamber two after all exact route deletions."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route


SKIPS = {42, 48, 62, 64, 66, 67, 68, 69, 70, 71, 72, 74}


def objects(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 9, 11, 12, 14), min_area=3
        )
        if blob.bbox[0] < 63
    )


def flipped(root):
    child = replay(root, route(), skips=SKIPS)
    child.step(*controls(child)[0])
    return child


def trace(root, name, action):
    child = flipped(root)
    print(name, "START", compact(child), "objects", objects(child))
    for count in range(1, 26):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print(
            name,
            count,
            compact(child),
            "terminal",
            bool(child.terminal()),
            "controls",
            controls(child),
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
