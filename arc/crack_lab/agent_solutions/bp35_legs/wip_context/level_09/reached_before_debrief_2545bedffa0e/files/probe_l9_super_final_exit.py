"""Take the final left-shaft exits after all commuting prefix deletions."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_fast_boost_endgame import FAST_SKIPS
from probe_l9_faster_prefix_deletions import TAIL
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_ultra_combined_deletions import BRIDGE, RETURN, ROOM
from probe_l9_ultra_cycle_stage import EXTRA_SKIPS


SUPER_SKIPS = FAST_SKIPS | EXTRA_SKIPS | ROOM | BRIDGE | RETURN


def super_boosted(root):
    child = replay(root, route(), skips=SUPER_SKIPS)
    for action in TAIL:
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def frontier(root):
    child = super_boosted(root)
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
    child.step(6, 21, 45)
    child.step(*controls(child)[0])
    for _ in range(3):
        child.step(6, 27, 33)
    child.step(6, 21, 27)
    child.step(3)
    child.step(*controls(child)[-1])
    for _ in range(3):
        child.step(6, 21, 33)
    return child


def probe(env):
    enter_level_9(env)
    child = frontier(env)
    name = sys.argv[1]
    report((name, 0), child)
    if name == "left":
        actions = ((6, 15, 39), 3, 4, 4, 4, 4, 4, 4)
    else:
        actions = (
            (6, 27, 39),
            4,
            (6, 33, 39),
            4,
            (6, 39, 39),
            4,
            (6, 45, 39),
            4,
            (6, 45, 33),
            (6, 45, 33),
            (6, 45, 33),
            (6, 45, 33),
            (6, 45, 33),
            (6, 45, 33),
        )
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
