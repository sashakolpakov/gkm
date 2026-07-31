"""Test each reachable landing column after the control-preserving wall flip."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_early21_c9_cycle_descent import wall_flipped
from probe_l9_early21_right import state
from probe_l9_route_deletions import enter_level_9


def probe(env):
    enter_level_9(env)
    root = wall_flipped(env, 1)
    state("ROOT", root)
    for left_steps in range(4):
        child = root.clone()
        for _ in range(left_steps):
            child.step(3)
        state((left_steps, "POSITION"), child)
        if child.terminal():
            continue
        x = 57 - 6 * left_steps
        child.step(6, x, 35)
        state((left_steps, x, "DROP"), child)
    interior = root.clone()
    for action in (3, 3, (6, 39, 27), 3):
        interior.step(*action) if isinstance(action, tuple) else interior.step(action)
        state(("INTERIOR", action), interior)
        if interior.terminal():
            break
    if not interior.terminal():
        for depth in range(1, 6):
            interior.step(6, 39, 35)
            state(("INTERIOR_DROP", depth), interior)
            if interior.terminal():
                break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
