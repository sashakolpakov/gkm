"""Continue down the uniquely safe sixth interior lane."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_second_descent import component_at
from probe_l9_second_gap_cross import enter_second_gap


def goals(env):
    return [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
        if blob.bbox[0] < 63
    ]


def enter_lane6_shelf(env):
    enter_second_gap(env, 6)
    for _ in range(7):
        env.step(6, 39, 35)


def probe(env):
    enter_lane6_shelf(env)
    print(
        "SHELF_DROP",
        compact(env),
        "under",
        component_at(env, 39, 35),
        "controls",
        controls(env),
        "goals",
        goals(env),
    )
    for depth in range(1, 26):
        under = component_at(env, 39, 35)
        if not under or under[0] != 15 or under[1] != 21:
            print("STOP", depth - 1, "under", under)
            return
        env.step(6, 39, 35)
        print(
            "DESCEND",
            depth,
            compact(env),
            "under",
            component_at(env, 39, 35),
            "controls",
            controls(env),
            "goals",
            goals(env),
        )
        if env.terminal() or int(env.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
