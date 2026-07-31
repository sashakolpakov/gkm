"""Test selector state 1 against every ring position and main orientation."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve
from probe_l6_exact_crossings import (
    CENTER,
    TO_CENTER,
    placement_label,
    placements_with_paths,
)
from probe_l6_right import (
    MAIN,
    SELECTOR,
    avatar_position,
    enter_right,
)


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    placements = placements_with_paths(enter_right(env, 3))
    for index, (placement, _) in enumerate(placements):
        hub = placement.clone()
        position = avatar_position(hub)
        if position != CENTER:
            hub.step(TO_CENTER[position])
        hub.step(*MAIN)
        hub.step(*SELECTOR)
        hub.step(*SELECTOR)
        for orientation in range(2):
            branch = hub.clone()
            if orientation:
                branch.step(4)
                branch.step(*MAIN)
                branch.step(3)
            before = avatar_position(branch)
            branch.step(*MAIN)
            after = avatar_position(branch)
            print(
                "PHASE1_ALL_RING", index, placement_label(placement),
                orientation, before, after,
                branch.levels_completed - base_level,
            )
            if after != before or branch.levels_completed > base_level:
                return


arena.run_program("dc22", observe)
