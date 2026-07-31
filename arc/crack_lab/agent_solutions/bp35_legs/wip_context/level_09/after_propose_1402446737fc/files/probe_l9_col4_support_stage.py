"""Stage a deeper column-four support before the otherwise lethal third drop."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_col5_yellow_trapdoor import col5_depth4
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


def frontier(root):
    child = col5_depth4(root)
    for action in (
        (6, 27, 35),
        (6, 27, 27),
        3,
        (6, 27, 35),
        (6, 27, 35),
    ):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def probe(env):
    enter_level_9(env)
    root = frontier(env)
    print("FRONTIER", compact(root), "objects", objects(root))
    for y in (39, 45, 51, 57):
        child = root.clone()
        child.step(6, 27, y)
        staged = compact(child)
        staged_objects = objects(child)
        if not child.terminal():
            child.step(6, 27, 35)
        print(
            "STAGE",
            y,
            "staged",
            staged,
            "staged_objects",
            staged_objects,
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
