"""Densely scan the D-pad at cargo terminal placements."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve
from probe_l6_cargo_top import CARGO_TOP_PATH
from probe_l6_dpad import dense_dpad_controls
from probe_l6_right import enter_right


def observe(env):
    solve.solve(env)
    node = enter_right(env, 3)
    for action in CARGO_TOP_PATH:
        if isinstance(action, tuple):
            node.step(*action)
        else:
            node.step(action)
    print("TERMINAL_DPAD_TOP", dense_dpad_controls(node))
    node.step(1)
    print("TERMINAL_TOP_EXIT", node.levels_completed, node.terminal())
    node.step(6, 50, 34)
    print("TERMINAL_TOP_EXIT_CARGO", node.levels_completed, node.terminal())


arena.run_program("dc22", observe)
