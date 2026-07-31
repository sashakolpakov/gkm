"""Test c9 descent after the shortcut's control-preserving up/down cycle."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import controls
from probe_l9_early21_c9_up import climbed
from probe_l9_early21_right import state
from probe_l9_route_deletions import enter_level_9


def wall_flipped(root, switch_index):
    child = climbed(root, switch_index)
    child.step(*controls(child)[0])
    return child


def probe(env):
    enter_level_9(env)
    for switch_index in (1, 2):
        child = wall_flipped(env, switch_index)
        state((switch_index, "WALL_FLIPPED"), child)
        for depth in range(1, 10):
            child.step(6, 57, 35)
            state((switch_index, "DROP", depth), child)
            if child.terminal() or int(child.levels_completed) >= 9:
                break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
