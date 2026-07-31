"""Cross the returned room's column-five catch gate into its right side."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_return_vertical_lanes import c2
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report


def probe(env):
    enter_level_9(env)
    child = c2(env)
    actions = (
        4,
        4,
        (6, 33, 39),
        4,
        (6, 39, 39),
        4,
        (6, 45, 39),
        4,
        4,
        (6, 45, 45),
        (6, 45, 51),
        (6, 45, 57),
        (6, 45, 51),
        (6, 45, 45),
        (6, 45, 33),
        (6, 45, 33),
        (6, 45, 33),
    )
    report(0, child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
