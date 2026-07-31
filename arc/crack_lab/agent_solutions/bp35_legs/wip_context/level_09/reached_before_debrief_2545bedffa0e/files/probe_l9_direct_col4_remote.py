"""Build the missing row-six lane-four catch from the direct yellow drop."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_direct_col4_depth5 import direct_col4_depth5
from probe_l9_route_deletions import enter_level_9


def objects(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 9, 11, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63 and (blob.color != 15 or blob.area == 21)
    )


def remote_drop(root):
    child = direct_col4_depth5(root)
    for action in ((6, 39, 39), (6, 33, 39), (6, 27, 35)):
        child.step(*action)
    return child


def probe(env):
    enter_level_9(env)
    child = remote_drop(env)
    print(
        "REMOTE_DROP",
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
    for action in ((6, 27, 35), 3, 4, (6, 27, 27), (6, 33, 27)):
        branch = child.clone()
        branch.step(*action) if isinstance(action, tuple) else branch.step(action)
        print(
            "ACTION",
            action,
            "terminal",
            bool(branch.terminal()),
            "level",
            int(branch.levels_completed) + 1,
            "controls",
            controls(branch),
            "state",
            compact(branch),
            "objects",
            objects(branch),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
