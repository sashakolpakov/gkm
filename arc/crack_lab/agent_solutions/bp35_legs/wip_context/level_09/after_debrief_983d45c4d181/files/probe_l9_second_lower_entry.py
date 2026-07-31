"""Enter each lane through one lower support staged before the second flip."""

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
            env.frame(), colors=(7, 8, 9, 11, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63 and (blob.color != 15 or blob.area == 21)
    )


def goals(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=2)
        if blob.bbox[0] < 63
    )


def chamber(root):
    return replay(root, route(), skips=SKIPS)


def staged_entry(root, col):
    child = chamber(root)
    x = 3 + 6 * col
    child.step(6, x, 51)
    child.step(*controls(child)[0])
    for _ in range(col - 1):
        child.step(4)
    child.step(6, x, 27)
    child.step(4)
    return child


def probe(env):
    enter_level_9(env)
    for col in range(2, 9):
        child = staged_entry(env, col)
        print(
            "ENTRY",
            col,
            "terminal",
            bool(child.terminal()),
            "level",
            int(child.levels_completed) + 1,
            "controls",
            controls(child),
            "goals",
            goals(child),
            "state",
            compact(child),
            "objects",
            objects(child),
        )
        if not child.terminal():
            below = child.clone()
            below.step(6, 3 + 6 * col, 33)
            print(
                "BELOW",
                col,
                "terminal",
                bool(below.terminal()),
                "level",
                int(below.levels_completed) + 1,
                "controls",
                controls(below),
                "goals",
                goals(below),
                "state",
                compact(below),
                "objects",
                objects(below),
            )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
