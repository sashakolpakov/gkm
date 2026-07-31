"""Climb through the lower wall gap after the depth-six aligned reversal."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report
from probe_l9_supported_final_alignment import aligned


def flipped(root, column):
    child = aligned(root, 6, column)
    child.step(*controls(child)[-1])
    return child


def probe(env):
    enter_level_9(env)
    for column in (5, 6, 7):
        child = flipped(env, column)
        report((column, 0), child)
        x = 3 + 6 * column
        for height in range(1, 17):
            child.step(6, x, 33)
            report((column, height), child)
            if child.terminal() or int(child.levels_completed) >= 9:
                break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
