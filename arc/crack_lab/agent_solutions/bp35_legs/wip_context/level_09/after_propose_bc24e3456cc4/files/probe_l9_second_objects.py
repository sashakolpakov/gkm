"""Classify every support in the second pre-flip control chamber."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact
from probe_l9_lane9_second_walk import enter_lane9_second_control


def probe(env):
    enter_lane9_second_control(env)
    blobs = connected_components(
        env.frame(), colors=(7, 8, 12, 14, 15), min_area=2
    )
    print("CHAMBER", compact(env))
    print(
        "OBJECTS",
        [
            (blob.color, blob.bbox, blob.area, tuple(round(v) for v in blob.centroid))
            for blob in blobs
            if blob.bbox[0] < 63
        ],
    )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
