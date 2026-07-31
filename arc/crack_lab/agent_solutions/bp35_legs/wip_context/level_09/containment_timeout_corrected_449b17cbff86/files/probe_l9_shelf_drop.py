"""Test opening each full colour-12 support under the avatar."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_second_gap_cross import enter_second_gap


def objects(env):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63
    ]


def probe(env):
    base = env.clone()
    for col in (2, 3, 4, 5, 6, 8):
        child = base.clone()
        enter_second_gap(child, col)
        x = 3 + 6 * col
        for _ in range(6):
            child.step(6, x, 35)
        before = compact(child)
        child.step(6, x, 35)
        print(
            "DROP",
            col,
            "before",
            before,
            "after",
            compact(child),
            "terminal",
            bool(child.terminal()),
            "controls",
            controls(child),
            "objects",
            objects(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
