"""Inspect the lower-maze endgame reached by one fewer upper-climb turn."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


def objects(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 9, 11, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63 and (blob.color != 15 or blob.area == 21)
    )


def report(label, env):
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "level",
        int(env.levels_completed) + 1,
        "controls",
        controls(env),
        "state",
        compact(env),
        "objects",
        objects(env),
    )


def root_skip(root):
    return replay(root, route(), skips=SKIPS | {11})


def probe(env):
    enter_level_9(env)
    root = root_skip(env)
    report("ROOT", root)
    for control in controls(root):
        child = root.clone()
        child.step(*control)
        report(("FLIP", control), child)
        for action in (
            3,
            4,
            7,
            (6, 21, 33),
            (6, 21, 27),
            (6, 21, 39),
            (6, 15, 33),
            (6, 27, 33),
        ):
            branch = child.clone()
            branch.step(*action) if isinstance(action, tuple) else branch.step(action)
            report(("ACTION", control, action), branch)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
