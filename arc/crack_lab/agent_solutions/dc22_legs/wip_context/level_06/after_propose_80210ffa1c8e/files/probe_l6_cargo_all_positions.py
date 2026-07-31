"""Visit every reachable cargo placement in one continuous route."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve
from probe_l6_right import enter_right


MOVEMENT = {"U": 1, "D": 2, "L": 3, "R": 4}
INVERSE = {"U": 2, "D": 1, "L": 4, "R": 3}
CONTROL = {
    "U": (6, 50, 34),
    "D": (6, 50, 40),
    "L": (6, 46, 36),
    "R": (6, 54, 36),
}
ROUTE = "RL" + "DLRRLU" + "LLLLRRRR" + "URUULLUUU" + "DDDRRDDLD"


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    node = enter_right(env, 3)
    current = None
    for index, target in enumerate(ROUTE, start=1):
        if current is not None:
            node.step(INVERSE[current])
        node.step(MOVEMENT[target])
        node.step(*CONTROL[target])
        current = target
        if node.levels_completed > base_level:
            print("CARGO_ALL_WIN", index, ROUTE[:index])
            return
    print("CARGO_ALL_NO_WIN", len(ROUTE), node.levels_completed)


arena.run_program("dc22", observe)
