"""Compare column-five endpoints from all four safe staged landing heights."""

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


def probe(env):
    enter_level_9(env)
    for landing_y in (39, 45, 51, 57):
        child = replay(env, route(), skips=SKIPS)
        child.step(*controls(child)[0])
        child.step(6, 51, landing_y)
        for _ in range(9):
            child.step(4)
        for col in range(8, 4, -1):
            child.step(6, 3 + 6 * col, 27)
            child.step(3)
        print(
            "LANDING",
            landing_y,
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


if __name__ == "__main__":
    arena.run_program("bp35", probe)
