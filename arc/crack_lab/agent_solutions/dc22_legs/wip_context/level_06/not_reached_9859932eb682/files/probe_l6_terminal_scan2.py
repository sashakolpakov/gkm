"""Two-pixel lattice scan for controls enabled at cargo-top state."""
from collections import defaultdict
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_cargo_top import CARGO_TOP_PATH
from probe_l6_right import enter_right


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    node = enter_right(env, 3)
    for action in CARGO_TOP_PATH:
        if isinstance(action, tuple):
            node.step(*action)
        else:
            node.step(action)
    before = perception.arr(node.frame()).copy()
    effects = defaultdict(list)
    for y in range(1, 64, 2):
        for x in range(1, 64, 2):
            branch = node.clone()
            branch.step(6, x, y)
            if branch.levels_completed > base_level:
                print("TERMINAL_SCAN_WIN", (x, y))
                return
            delta = perception.frame_delta(before, branch.frame())
            world_samples = [
                sample for sample in delta["samples"] if sample[0] < 63
            ]
            if world_samples:
                effects[(delta["count"], delta["bbox"])].append((x, y))
    for signature, points in sorted(effects.items()):
        print("TERMINAL_SCAN_EFFECT", signature, points)


arena.run_program("dc22", observe)
