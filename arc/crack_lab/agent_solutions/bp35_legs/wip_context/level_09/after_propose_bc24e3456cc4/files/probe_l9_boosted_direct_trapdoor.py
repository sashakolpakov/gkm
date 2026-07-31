"""Fall through the boosted yellow trapdoor after only one gravity reversal."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_oneflip_trapdoors import yellows
from probe_l9_preserve_boost import boosted
from probe_l9_route_deletions import enter_level_9


def probe(env):
    enter_level_9(env)
    child = boosted(env)
    child.step(*controls(child)[0])
    report("FIRST_FLIP", child)
    child.step(3)
    child.step(3)
    report("ALIGNED", child)
    left_yellow = min(yellows(child), key=lambda action: action[1])
    child.step(*left_yellow)
    report("DEEP", child)
    for index in range(1, 4):
        child.step(4)
        report(("RIGHT", index), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return
    visible = controls(child)
    if visible:
        child.step(*visible[-1])
        report("WALL_FLIP", child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
