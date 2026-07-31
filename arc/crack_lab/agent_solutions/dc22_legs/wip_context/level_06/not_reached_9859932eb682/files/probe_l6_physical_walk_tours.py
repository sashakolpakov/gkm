"""Walk the physical component under every ring/A/B configuration."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_exact_crossings import (
    horizontal_entry,
    placement_label,
    placements_with_paths,
)
from probe_l6_global_walk_tours import exits, walk_tour
from probe_l6_right import MAIN, TOP, enter_right


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    placements = placements_with_paths(enter_right(env, 3))
    checked = 0
    walked = 0
    for index, (placement, _) in enumerate(placements):
        physical = horizontal_entry(placement)
        for orientation in range(2):
            oriented = physical.clone()
            if orientation:
                oriented.step(*MAIN)
            for bridge_phase in range(6):
                staged = oriented.clone()
                for _ in range(bridge_phase):
                    staged.step(*TOP)
                checked += 1
                tour = walk_tour(staged)
                for step_index, action in enumerate(tour, start=1):
                    staged.step(action)
                    walked += 1
                    if staged.levels_completed > base_level:
                        print(
                            "PHYSICAL_TOUR_WIN", (
                                index, placement_label(placement),
                                orientation, bridge_phase,
                            ),
                            "tour_step", step_index,
                            "tour_prefix", tour[:step_index],
                            "checked", checked, "walked", walked,
                            flush=True,
                        )
                        return
                visible = exits(staged)
                if visible:
                    print(
                        "PHYSICAL_TOUR_EXIT", (
                            index, placement_label(placement),
                            orientation, bridge_phase,
                        ),
                        visible, "checked", checked, flush=True,
                    )
                    return
        print(
            "PHYSICAL_TOUR_DONE", index,
            "checked", checked, "walked", walked, flush=True,
        )
    print(
        "PHYSICAL_TOUR_NO_WIN", checked, walked, flush=True,
    )


arena.run_program("dc22", observe)
