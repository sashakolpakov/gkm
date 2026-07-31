"""Inspect and locally probe the post-shelf lane-six supports."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact
from probe_l9_lane6_continue import enter_lane6_shelf


def objects(env):
    return [
        (blob.color, blob.bbox, blob.area, tuple(round(v) for v in blob.centroid))
        for blob in connected_components(
            env.frame(), colors=(7, 8, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63
    ]


def probe(env):
    enter_lane6_shelf(env)
    print("ENTRY", compact(env), "objects", objects(env))
    for x, y in ((51, 27), (45, 27), (51, 39), (57, 27)):
        child = env.clone()
        child.step(6, x, y)
        print(
            "CLICK",
            (x, y),
            compact(child),
            "terminal",
            bool(child.terminal()),
            "objects",
            objects(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
