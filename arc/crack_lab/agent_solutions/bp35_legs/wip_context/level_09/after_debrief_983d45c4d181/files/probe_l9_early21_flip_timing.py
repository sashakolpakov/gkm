"""Map second-flip heights made possible by the shortcut's surplus controls."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_early21_c9_up import c9
from probe_l9_route_deletions import enter_level_9


def probe(env):
    enter_level_9(env)
    up_root = c9(env)
    up_root.step(*controls(up_root)[-1])
    for height in range(8):
        if height:
            up_root.step(6, 57, 33)
        visible = controls(up_root)
        report((height, "UP", tuple(visible)), up_root)
        for switch_index, switch in enumerate(visible):
            child = up_root.clone()
            child.step(*switch)
            report((height, switch_index, switch, "DOWN"), child)
            if child.terminal():
                continue
            drop = child.clone()
            drop.step(6, 57, 35)
            report((height, switch_index, "DROP"), drop)
            left = child.clone()
            left.step(3)
            report((height, switch_index, "LEFT"), left)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
