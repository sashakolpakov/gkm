import numpy as np

import gkm_try as harness
from perception import color_counts


def do(node, action, count):
    for _ in range(count):
        node.step(action)


def probe(env):
    harness.resumed_solve(env)
    for left_turns, left_dx, right_turns, right_dx in (
        (0, 10, 1, 1),
        (0, 10, 3, 2),
        (2, 12, 1, 1),
        (2, 12, 3, 2),
    ):
        node = env.clone()
        do(node, 5, 4)
        do(node, 1, 3)
        do(node, 4, 6)
        node.step(6, 54, 6)
        do(node, 5, 3)
        node.step(6, 5, 38)
        do(node, 5, left_turns)
        do(node, 1, 10)
        do(node, 4, left_dx)
        node.step(6, 47, 47)
        do(node, 5, right_turns)
        do(node, 1, 10)
        do(node, 4, right_dx)
        frame = np.asarray(node.frame())
        print("FINAL", (left_turns, left_dx, right_turns, right_dx),
              node.levels_completed, color_counts(frame),
              int(np.count_nonzero(frame[1:] != 15)))


harness.A.run_program("cn04", probe)
