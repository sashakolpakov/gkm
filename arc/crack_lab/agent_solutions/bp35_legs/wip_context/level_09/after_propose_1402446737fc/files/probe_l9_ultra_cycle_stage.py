"""Stage the second reversal after all exact boosted-prefix deletions."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_fast_boost_endgame import FAST_SKIPS
from probe_l9_faster_prefix_deletions import TAIL
from probe_l9_route_deletions import enter_level_9, replay, route


EXTRA_SKIPS = {21, 22, 23, 24, 25, 26, 73, 110}
ULTRA_SKIPS = FAST_SKIPS | EXTRA_SKIPS


def ultra_boosted(root):
    child = replay(root, route(), skips=ULTRA_SKIPS)
    for action in TAIL:
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def height_eight(root):
    child = ultra_boosted(root)
    child.step(*controls(child)[0])
    child.step(3)
    child.step(6, 21, 33)
    child.step(6, 21, 35)
    for _ in range(3):
        child.step(6, 21, 33)
    child.step(6, 27, 27)
    child.step(4)
    child.step(*controls(child)[-1])
    for _ in range(8):
        child.step(6, 27, 33)
    return child


def probe(env):
    enter_level_9(env)
    stage_y = int(sys.argv[1])
    switch_index = int(sys.argv[2])
    child = height_eight(env)
    child.step(6, 21, stage_y)
    report((stage_y, switch_index, "STAGED"), child)
    switch = controls(child)[switch_index]
    child.step(*switch)
    report((stage_y, switch_index, switch, "FLIP"), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
