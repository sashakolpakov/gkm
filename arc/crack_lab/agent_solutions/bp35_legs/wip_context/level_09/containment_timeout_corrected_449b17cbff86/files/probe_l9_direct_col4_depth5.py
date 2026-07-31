"""Inspect direct column four immediately after removing its yellow support."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


def direct_col4_depth5(root):
    child = replay(root, route(), skips=SKIPS)
    child.step(*controls(child)[0])
    for _ in range(4):
        child.step(4)
    for _ in range(5):
        child.step(6, 27, 35)
    return child


def probe(env):
    enter_level_9(env)
    child = direct_col4_depth5(env)
    print("STATE", compact(child))
    for color in (7, 8, 12, 14, 15):
        print(
            "COLOR",
            color,
            tuple(
                (blob.bbox, blob.area)
                for blob in connected_components(
                    child.frame(), colors=(color,), min_area=2
                )
                if blob.bbox[0] < 63
            ),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
