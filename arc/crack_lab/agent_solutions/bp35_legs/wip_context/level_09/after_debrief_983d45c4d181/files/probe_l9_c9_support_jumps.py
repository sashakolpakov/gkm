"""Use off cells in the visible support lattice as deeper column-nine landings."""

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


def row57_landing(root):
    child = replay(root, route(), skips=SKIPS)
    child.step(*controls(child)[0])
    child.step(6, 51, 57)
    for _ in range(9):
        child.step(4)
    return child


def run(root, name, target):
    child = row57_landing(root)
    print(name, "START", compact(child), "objects", objects(child))
    child.step(6, *target)
    print(
        name,
        "TOGGLE",
        target,
        compact(child),
        "terminal",
        bool(child.terminal()),
        "objects",
        objects(child),
    )
    if not child.terminal():
        child.step(6, 57, 33)
        print(
            name,
            "FALL",
            compact(child),
            "terminal",
            bool(child.terminal()),
            "controls",
            controls(child),
            "objects",
            objects(child),
        )


def probe(env):
    enter_level_9(env)
    for y in (45, 51, 57):
        run(env, f"C9_R{(y - 3) // 6}", (57, y))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
