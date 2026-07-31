"""Grow the remote c7 catch down and left to make a safe c5 side-step."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_handoff_clickmap import avatar
from probe_l9_presecond_top_c4_climb import goals
from probe_l9_presecond_upper_landing_actions import upper_landing
from probe_l9_right_trap_stage import full_catches
from probe_l9_route_deletions import enter_level_9


def state(label, child):
    report(label, child)
    print(
        "STATE",
        label,
        "terminal",
        bool(child.terminal()),
        "avatar",
        avatar(child),
        "controls",
        controls(child),
        "goals",
        goals(child),
        "full",
        full_catches(child),
        flush=True,
    )


def staged_c5(root):
    child = upper_landing(root)
    for y in (3, 9, 15, 21, 27, 33):
        child.step(6, 45, y)
    child.step(6, 39, 33)
    child.step(6, 33, 33)
    return child


def probe(env):
    enter_level_9(env)
    child = upper_landing(env)
    state("ROOT", child)
    for index, action in enumerate(
        (
            (6, 45, 3),
            (6, 45, 9),
            (6, 45, 15),
            (6, 45, 21),
            (6, 45, 27),
            (6, 45, 33),
            (6, 39, 33),
            (6, 33, 33),
            (6, 33, 39),
            (4,),
        ),
        1,
    ):
        child.step(*action)
        state((index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
