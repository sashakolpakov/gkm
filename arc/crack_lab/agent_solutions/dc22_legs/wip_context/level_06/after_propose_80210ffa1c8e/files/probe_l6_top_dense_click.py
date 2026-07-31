"""Densely probe the occupied top endpoint for coordinate interaction."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import MAIN, SELECTOR, avatar_position, enter_right


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    node = enter_right(env, 3)
    node.step(*MAIN)
    for _ in range(3):
        node.step(*SELECTOR)
    node.step(*MAIN)
    print("TOP_DENSE_START", avatar_position(node))
    before = perception.arr(node.frame()).copy()
    for y in range(4, 12):
        for x in range(4, 12):
            branch = node.clone()
            branch.step(6, x, y)
            if branch.levels_completed > base_level:
                print("TOP_DENSE_WIN", (x, y))
                return
            delta = perception.frame_delta(before, branch.frame())
            samples = [
                sample for sample in delta["samples"] if sample[0] < 63
            ]
            if samples:
                print("TOP_DENSE_EFFECT", (x, y), samples)
    print("TOP_DENSE_NONE")


arena.run_program("dc22", observe)
