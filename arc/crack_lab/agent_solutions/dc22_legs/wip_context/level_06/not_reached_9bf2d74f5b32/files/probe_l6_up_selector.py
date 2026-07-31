"""Test the cargo lift under each selector context."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import (
    SELECTOR,
    enter_right,
    movement_reach,
)


UP_CONTROL = (6, 50, 34)


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    activated = enter_right(env, 3)
    reached, _ = movement_reach(activated)
    for action in reached[(56, 34)]:
        activated.step(action)
    for selector_offset in range(4):
        branch = activated.clone()
        for _ in range(selector_offset):
            branch.step(*SELECTOR)
        branch.step(*UP_CONTROL)
        region = perception.arr(branch.frame())[8:40, 14:32]
        rings = [
            blob.bbox
            for blob in perception.connected_components(
                branch.frame(), colors=(8,), min_area=20
            )
            if blob.bbox[1] < 32
        ]
        print(
            "UP_SELECTOR", selector_offset, rings,
            perception.color_counts(region),
            branch.levels_completed - base_level,
        )


arena.run_program("dc22", observe)
