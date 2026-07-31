"""Exit the early column-eight flip through the exterior column-nine shaft."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_lane7_early_flip_local import objects
from probe_l9_pre_deep_flip_lanes import before_flip
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


def root_lane8(root):
    child = before_flip(root)
    child.step(3)
    child.step(*controls(child)[0])
    return child


def probe(env):
    enter_level_9(env)
    root = root_lane8(env)
    report("ROOT", root)
    for action in (
        3,
        4,
        7,
        (6, 57, 33),
        (6, 57, 39),
        (6, 57, 45),
        (6, 51, 33),
        (6, 51, 39),
        (6, 51, 45),
    ):
        child = root.clone()
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report(("ACTION", action), child)
        if action == 4 and not child.terminal():
            for index in range(1, 9):
                child.step(6, 57, 33)
                report(("C9_WAIT", index), child)
                if child.terminal() or int(child.levels_completed) >= 9:
                    break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
