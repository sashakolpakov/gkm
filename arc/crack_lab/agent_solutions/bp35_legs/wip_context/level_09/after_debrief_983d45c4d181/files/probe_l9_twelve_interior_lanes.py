"""Descend directly in each supported interior lane after the second flip."""

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


def probe(env):
    enter_level_9(env)
    for col in range(2, 9):
        child = flipped(env)
        for _ in range(col):
            child.step(4)
        print("LANE", col, "START", compact(child), "objects", objects(child))
        x = 3 + 6 * col
        for depth in range(1, 16):
            child.step(6, x, 35)
            visible = objects(child)
            if (
                child.terminal()
                or int(child.levels_completed) >= 9
                or controls(child)
                or any(item[0] == 7 for item in visible)
                or depth in (4, 8, 12)
            ):
                print(
                    "LANE",
                    col,
                    "DEPTH",
                    depth,
                    "terminal",
                    bool(child.terminal()),
                    "level",
                    int(child.levels_completed) + 1,
                    "controls",
                    controls(child),
                    "state",
                    compact(child),
                    "objects",
                    visible,
                )
            if child.terminal() or int(child.levels_completed) >= 9:
                break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
