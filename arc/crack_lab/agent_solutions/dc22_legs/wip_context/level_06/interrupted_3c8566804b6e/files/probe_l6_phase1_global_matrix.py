"""Test the missing selector destination across all visible global phases."""
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
from probe_l6_right import MAIN, SELECTOR, TOP, avatar_position, enter_right


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    placements = placements_with_paths(enter_right(env, 3))
    checked = 0
    for index, (placement, _) in enumerate(placements):
        centered = placement.clone()
        position = avatar_position(centered)
        if position != CENTER:
            centered.step(TO_CENTER[position])
        for orientation in range(2):
            oriented = centered.clone()
            if orientation:
                oriented.step(4)
                oriented.step(*MAIN)
                oriented.step(3)
            for bridge_phase in range(6):
                staged = oriented.clone()
                for _ in range(bridge_phase):
                    staged.step(*TOP)
                staged.step(*MAIN)
                hub = avatar_position(staged)
                staged.step(*SELECTOR)
                staged.step(*SELECTOR)
                before = avatar_position(staged)
                staged.step(*MAIN)
                after = avatar_position(staged)
                checked += 1
                if after != before or staged.levels_completed > base_level:
                    print(
                        "PHASE1_GLOBAL_HIT", index,
                        placement_label(placement),
                        "orientation", orientation,
                        "bridge", bridge_phase,
                        "hub", hub, "before", before, "after", after,
                        "level", staged.levels_completed,
                        "checked", checked, flush=True,
                    )
                    return
        print("PHASE1_GLOBAL_DONE", index, checked, flush=True)
    print("PHASE1_GLOBAL_NO_HIT", checked, flush=True)


arena.run_program("dc22", observe)
