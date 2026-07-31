"""Resolve full catches versus small hazards at the column-six frontier."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_col6_depth7_actions import col6_depth7
from probe_l9_control_row import compact
from probe_l9_route_deletions import enter_level_9


def probe(env):
    enter_level_9(env)
    child = col6_depth7(env)
    print("STATE", compact(child))
    for color in (12, 14, 15):
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
