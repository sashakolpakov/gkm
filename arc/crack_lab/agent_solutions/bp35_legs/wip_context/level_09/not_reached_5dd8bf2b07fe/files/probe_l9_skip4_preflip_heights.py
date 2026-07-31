"""Stage vertical height before the four-skip flip, then replay the goal shortcut."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9, route, step
from probe_l9_skip4_switch_choices import before_flip, report


UP = (6, 27, 33)


def probe(env):
    enter_level_9(env)
    for height in range(7):
        child = before_flip(env)
        for _ in range(height):
            child.step(*UP)
            if child.terminal():
                break
        report((height, "STAGED"), child)
        visible = controls(child)
        if child.terminal() or not visible:
            continue
        child.step(*visible[0])
        report((height, "FLIP"), child)
        for index in range(28, 34):
            step(child, route()[index][1])
            if child.terminal() or int(child.levels_completed) >= 9:
                break
        report((height, "WALL"), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
