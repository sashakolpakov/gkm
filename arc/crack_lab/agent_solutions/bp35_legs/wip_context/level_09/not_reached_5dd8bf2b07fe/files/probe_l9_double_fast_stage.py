"""Test the visible support-edge and block affordances after flip two."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route


def objects(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 12, 14, 15), min_area=3
        )
        if blob.bbox[0] < 63
    )


def flipped(root):
    child = replay(root, route(), skips=(42, 48))
    child.step(*controls(child)[0])
    return child


def run(root, name, prefix):
    child = flipped(root)
    print(name, "START", compact(child), "objects", objects(child))
    for action in prefix:
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print(name, "STAGE", action, compact(child), "objects", objects(child))
        if child.terminal():
            return
    for count in range(1, 13):
        child.step(4)
        print(
            name,
            "RIGHT",
            count,
            compact(child),
            "terminal",
            bool(child.terminal()),
            "objects",
            objects(child),
        )
        if child.terminal() or int(child.levels_completed) >= 9:
            return


def probe(env):
    enter_level_9(env)
    run(env, "EDGE_CATCH", [(6, 51, 33)])
    run(env, "BLOCK", [(6, 27, 59)])
    run(env, "EDGE_THEN_BLOCK", [(6, 51, 33), (6, 27, 59)])


if __name__ == "__main__":
    arena.run_program("bp35", probe)
