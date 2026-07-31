"""Extend deeper row supports into column nine and compare free-fall landings."""

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


def flipped(root):
    child = replay(root, route(), skips=SKIPS)
    child.step(*controls(child)[0])
    return child


def run(root, name, rows):
    child = flipped(root)
    for y in rows:
        child.step(6, 51, y)
    for _ in range(9):
        child.step(4)
        if child.terminal():
            break
    print(
        name,
        "rows",
        rows,
        "terminal",
        bool(child.terminal()),
        "state",
        compact(child),
        "controls",
        controls(child),
        "objects",
        objects(child),
    )
    if not child.terminal():
        below = child.clone()
        below.step(6, 57, 33)
        print(
            name,
            "BELOW",
            "terminal",
            bool(below.terminal()),
            "state",
            compact(below),
            "controls",
            controls(below),
            "objects",
            objects(below),
        )


def probe(env):
    enter_level_9(env)
    for y in (39, 45, 51, 57):
        run(env, f"ROW{y}", (y,))
    run(env, "STACK39_45", (39, 45))
    run(env, "STACK45_51_57", (45, 51, 57))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
