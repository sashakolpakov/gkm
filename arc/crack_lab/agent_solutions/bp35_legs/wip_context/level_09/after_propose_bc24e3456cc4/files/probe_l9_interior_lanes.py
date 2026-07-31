"""Compare safe descent depth in each lane after crossing the second gap."""

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


def probe(env):
    base = env.clone()
    for col in range(2, 9):
        child = base.clone()
        enter_second_gap(child, col)
        x = 3 + 6 * col
        seen_goal = None
        depth = 0
        for depth in range(1, 26):
            under = component_at(child, x, 35)
            if not under or under[0] != 15 or under[1] != 21:
                depth -= 1
                break
            child.step(6, x, 35)
            visible = goals(child)
            if visible and seen_goal is None:
                seen_goal = (depth, visible)
            if child.terminal() or int(child.levels_completed) >= 9:
                break
        print(
            "LANE",
            col,
            "depth",
            depth,
            "terminal",
            bool(child.terminal()),
            "level",
            int(child.levels_completed) + 1,
            "under",
            component_at(child, x, 35),
            "controls",
            controls(child),
            "seen_goal",
            seen_goal,
            "state",
            compact(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
