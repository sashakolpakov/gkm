"""Stage the ceiling catch before climbing from the final column-seven corridor."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_right_trap_stage import full_catches
from probe_l9_route_deletions import enter_level_9
from probe_l9_super_final_exit import frontier


TO_C7 = (
    (6, 27, 39),
    4,
    (6, 33, 39),
    4,
    (6, 39, 39),
    4,
    (6, 45, 39),
    4,
)

PLANS = {
    "none": (),
    "two_above": ((6, 45, 27), (6, 45, 33)),
    "three_above": ((6, 45, 21), (6, 45, 33)),
    "above_pair": ((6, 45, 27), (6, 45, 21), (6, 45, 33)),
    "below": ((6, 45, 45), (6, 45, 33)),
    "upper_left": ((6, 39, 33), (6, 45, 33)),
}


def probe(env):
    enter_level_9(env)
    child = frontier(env)
    for action in TO_C7:
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    name = sys.argv[1]
    report((name, 0), child)
    print("FULL", full_catches(child), flush=True)
    for index, action in enumerate(PLANS[name], 1):
        child.step(*action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
