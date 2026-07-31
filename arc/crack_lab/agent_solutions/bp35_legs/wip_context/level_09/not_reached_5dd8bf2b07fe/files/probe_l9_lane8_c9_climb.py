"""Climb the staged exterior column-nine shaft after the early flip."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_lane8_early_flip_exit import root_lane8
from probe_l9_route_deletions import enter_level_9


def goals(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=2)
        if blob.bbox[0] < 63
    )


def staged_c9(root):
    child = root_lane8(root)
    for action in ((6, 57, 45), (6, 57, 39), 4):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def probe(env):
    enter_level_9(env)
    child = staged_c9(env)
    print(
        "START",
        "terminal",
        bool(child.terminal()),
        "controls",
        controls(child),
        "goals",
        goals(child),
        "state",
        compact(child),
    )
    for index in range(1, 28):
        child.step(6, 57, 33)
        print(
            "CLIMB",
            index,
            "terminal",
            bool(child.terminal()),
            "level",
            int(child.levels_completed) + 1,
            "controls",
            controls(child),
            "goals",
            goals(child),
            "state",
            compact(child),
        )
        if child.terminal() or int(child.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
