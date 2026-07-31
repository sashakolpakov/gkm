"""Return from the lower gap to column two, then enter the known goal corridor."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report
from probe_l9_supported_gap_climb import flipped


def probe(env):
    enter_level_9(env)
    child = flipped(env, 6)
    actions = (
        3,
        3,
        3,
        (6, 15, 39),
        3,
        (6, 15, 33),
        4,
        4,
        4,
        4,
        4,
        4,
    )
    report(0, child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
