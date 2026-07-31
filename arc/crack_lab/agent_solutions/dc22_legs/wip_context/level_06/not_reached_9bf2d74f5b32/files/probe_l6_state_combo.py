"""Exhaust finite control states after satisfying the visible right goals."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve
from probe_l6_right import MAIN, SELECTOR, TOP, enter_right


UP_CONTROL = (6, 50, 34)


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    base = enter_right(env, 3)
    base.step(1)
    base.step(*UP_CONTROL)
    wins = []
    for bridge_steps in range(6):
        for main_steps in range(2):
            for selector_steps in range(4):
                branch = base.clone()
                for _ in range(bridge_steps):
                    branch.step(*TOP)
                for _ in range(main_steps):
                    branch.step(*MAIN)
                for _ in range(selector_steps):
                    branch.step(*SELECTOR)
                if branch.levels_completed > base_level:
                    wins.append(
                        (bridge_steps, main_steps, selector_steps)
                    )
    print("STATE_COMBO_WINS", wins)


arena.run_program("dc22", observe)
