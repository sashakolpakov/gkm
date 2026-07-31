"""Test staging each wheel's unique color-11 token at the central connector."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from plan_l6 import BOARDS, read_board
from probe_l6 import CONTROLS
from solve import solve


PATH = (
    [CONTROLS[0]] * 2 + [CONTROLS[1]] * 2 +
    [CONTROLS[4]] * 4 + [CONTROLS[6]] * 2 +
    [CONTROLS[2]] * 6 + [CONTROLS[3]] * 2 +
    [CONTROLS[5]]
)


def run(env):
    solve(env)
    base = env.levels_completed
    before = tuple(read_board(np.asarray(env.frame()), spec)
                   for spec in BOARDS)
    print("green_indices", tuple(state.index(11) for state in before),
          "buffers", tuple(state[24] for state in before),
          "path_len", len(PATH))
    clone = env.clone()
    for n, point in enumerate(PATH, 1):
        clone.step(6, *point)
        if clone.levels_completed > base:
            print("REWARD", clone.levels_completed, "at", n)
            print("PATH", PATH[:n])
            return
    after = tuple(read_board(np.asarray(clone.frame()), spec)
                  for spec in BOARDS)
    print("NO_REWARD", clone.levels_completed,
          "buffers", tuple(state[24] for state in after))


if __name__ == "__main__":
    A.run_program("lp85", run)
