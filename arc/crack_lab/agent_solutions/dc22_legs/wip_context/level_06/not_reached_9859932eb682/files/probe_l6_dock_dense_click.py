"""Densely scan coordinate interaction at the exactly centered ring dock."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_exact_crossings import placements_with_paths
from probe_l6_right import enter_right


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    placements = placements_with_paths(enter_right(env, 3))
    for index in (11, 13):
        node = placements[index][0]
        before = perception.arr(node.frame()).copy()
        effects = []
        for y in range(28, 41):
            for x in range(4, 20):
                branch = node.clone()
                branch.step(6, x, y)
                if branch.levels_completed > base_level:
                    print("DOCK_DENSE_WIN", index, (x, y))
                    return
                delta = perception.frame_delta(before, branch.frame())
                samples = [
                    sample for sample in delta["samples"]
                    if sample[0] < 63
                ]
                if samples:
                    effects.append(((x, y), delta["count"], delta["bbox"]))
        print("DOCK_DENSE_EFFECTS", index, effects)


arena.run_program("dc22", observe)
