"""Descend the second edge shaft after staging the gap bridge."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import cell_symbol
from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_gap_flip import enter_gap_landing


def component_at(env, x, y):
    for blob in connected_components(
        env.frame(), colors=(7, 8, 14, 15), min_area=3
    ):
        y0, x0, y1, x1 = blob.bbox
        if y0 <= y <= y1 and x0 <= x <= x1:
            return blob.color, blob.area, blob.bbox
    return None


def goals(env):
    return [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
        if blob.bbox[0] < 63
    ]


def probe(env):
    enter_gap_landing(env)
    print(
        "LANDING",
        compact(env),
        "under",
        component_at(env, 3, 35),
        "controls",
        controls(env),
        "goals",
        goals(env),
    )
    for depth in range(1, 31):
        under = component_at(env, 3, 35)
        if not under or under[0] != 15 or under[1] != 21:
            print("STOP", depth - 1, "under", under)
            return
        env.step(6, 3, 35)
        print(
            "DESCEND",
            depth,
            compact(env),
            "under",
            component_at(env, 3, 35),
            "controls",
            controls(env),
            "goals",
            goals(env),
            "cell",
            cell_symbol(env.frame()[35][3]),
        )
        if env.terminal() or int(env.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
