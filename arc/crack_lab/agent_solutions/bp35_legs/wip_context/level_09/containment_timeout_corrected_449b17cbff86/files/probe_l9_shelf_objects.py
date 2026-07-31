"""List object areas around the first interior shelf."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact
from probe_l9_second_gap_cross import enter_second_gap


def probe(env):
    enter_second_gap(env, 2)
    for _ in range(6):
        env.step(6, 15, 35)
    blobs = connected_components(
        env.frame(), colors=(7, 8, 12, 14, 15), min_area=2
    )
    print("SHELF", compact(env))
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
