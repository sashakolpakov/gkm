"""Climb the c5 column after completing the remotely staged side handoff."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_handoff_clickmap import avatar
from probe_l9_presecond_top_c4_climb import goals
from probe_l9_presecond_upper_stage_right import staged_c5
from probe_l9_right_trap_stage import full_catches
from probe_l9_route_deletions import enter_level_9


def entered_c5(root):
    child = staged_c5(root)
    child.step(6, 33, 39)
    child.step(4)
    return child


def probe(env):
    enter_level_9(env)
    child = entered_c5(env)
    for height in range(11):
        report(("HEIGHT", height), child)
        print(
            "STATE",
            height,
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
        if child.terminal() or int(child.levels_completed) >= 9:
            return
        child.step(6, 33, 33)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
