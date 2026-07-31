"""Exhaust control-state commits at the cargo's top terminal."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve
from probe_l6_cargo_top import CARGO_TOP_PATH
from probe_l6_right import MAIN, SELECTOR, TOP, enter_right


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    terminal = enter_right(env, 3)
    for action in CARGO_TOP_PATH:
        if isinstance(action, tuple):
            terminal.step(*action)
        else:
            terminal.step(action)
    wins = []
    for bridge_steps in range(6):
        for main_steps in range(2):
            for selector_steps in range(4):
                branch = terminal.clone()
                for _ in range(bridge_steps):
                    branch.step(*TOP)
                    if branch.levels_completed > base_level:
                        wins.append((bridge_steps, main_steps, selector_steps, "top"))
                        break
                for _ in range(main_steps):
                    branch.step(*MAIN)
                    if branch.levels_completed > base_level:
                        wins.append((bridge_steps, main_steps, selector_steps, "main"))
                        break
                for _ in range(selector_steps):
                    branch.step(*SELECTOR)
                    if branch.levels_completed > base_level:
                        wins.append((bridge_steps, main_steps, selector_steps, "selector"))
                        break
    print("CARGO_TOP_COMBO_WINS", wins)


arena.run_program("dc22", observe)
