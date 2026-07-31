"""Climb through an adjacent prepared shaft from the early-flip left wall."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_lane7_cross_left import at_left_wall
from probe_l9_lane7_early_flip_local import objects
from probe_l9_route_deletions import enter_level_9


def report(label, env):
    goals = tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=2)
        if blob.bbox[0] < 63
    )
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "level",
        int(env.levels_completed) + 1,
        "controls",
        controls(env),
        "goals",
        goals,
        "state",
        compact(env),
        "objects",
        objects(env),
    )


def first_climb(root):
    child = at_left_wall(root)
    for action in ((6, 21, 33), (6, 21, 39), 4):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def probe(env):
    enter_level_9(env)
    child = at_left_wall(env)
    report("START", child)
    for index, action in enumerate(((6, 21, 33), (6, 21, 39), 4), 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report(("STEP", index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return
    for action in (
        3,
        4,
        7,
        (6, 21, 33),
        (6, 21, 39),
        (6, 21, 45),
        (6, 15, 33),
        (6, 27, 33),
        (6, 27, 39),
        (6, 33, 33),
        (6, 33, 39),
    ):
        branch = child.clone()
        branch.step(*action) if isinstance(action, tuple) else branch.step(action)
        report(("ACTION", action), branch)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
