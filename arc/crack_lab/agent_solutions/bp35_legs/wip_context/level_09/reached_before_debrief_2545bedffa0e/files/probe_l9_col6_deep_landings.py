"""Push the future lane-six landing deeper before entering it once."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_col5_depth6_actions import col5_depth6
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9


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
    stage_rows = (33, 39, 45, 51, 57)
    for count in range(1, len(stage_rows) + 1):
        child = col5_depth6(env)
        for y in stage_rows[:count]:
            child.step(6, 39, y)
        child.step(6, 39, 27)
        child.step(4)
        print(
            "STAGES",
            count,
            stage_rows[:count],
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
