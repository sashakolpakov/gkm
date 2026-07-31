"""Use the row-nine landing to hand off left before the interior walls begin."""

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
    child = replay(env, route(), skips=SKIPS)
    child.step(*controls(child)[0])
    child.step(6, 51, 57)
    for _ in range(9):
        child.step(4)
    print("LAND", compact(child), "objects", objects(child))
    for col in range(8, 4, -1):
        action = (6, 3 + 6 * col, 27)
        child.step(*action)
        print(
            "OPEN",
            col,
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
        child.step(3)
        print(
            "LEFT",
            col,
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


if __name__ == "__main__":
    arena.run_program("bp35", probe)
