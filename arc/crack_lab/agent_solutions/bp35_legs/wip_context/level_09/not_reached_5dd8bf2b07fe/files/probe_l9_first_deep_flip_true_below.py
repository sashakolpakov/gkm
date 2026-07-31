"""Test the actual row-seven catches below each intermediate return lane."""

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
    root = replay(env, route()[:111], skips=SKIPS)
    for lefts in range(1, 10):
        child = root.clone()
        for _ in range(lefts):
            child.step(3)
        col = 9 - lefts
        x = 3 + 6 * col
        before = compact(child)
        child.step(6, x, 45)
        print(
            "LANE",
            col,
            "lefts",
            lefts,
            "before",
            before,
            "terminal",
            bool(child.terminal()),
            "controls",
            controls(child),
            "state",
            compact(child),
            "objects",
            objects(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
