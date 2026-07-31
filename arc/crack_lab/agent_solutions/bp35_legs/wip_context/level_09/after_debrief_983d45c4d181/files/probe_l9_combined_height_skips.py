"""Combine upper-climb shortcuts with omissions in the repeated lane-nine drop."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


def boxes(env, color):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(color,), min_area=2)
        if blob.bbox[0] < 63
    )


def probe(env):
    enter_level_9(env)
    actions = route()
    for upper in (3, 4):
        for drops in range(8):
            omitted = (
                SKIPS
                | set(range(11, 11 + upper))
                | set(range(103, 103 + drops))
            )
            child = replay(env, actions, skips=omitted)
            print(
                "END",
                "upper",
                upper,
                "drops",
                drops,
                "applied",
                len(actions) - len(omitted),
                "terminal",
                bool(child.terminal()),
                "levels",
                int(child.levels_completed),
                "controls",
                controls(child),
                "goals",
                boxes(child, 7),
                "avatar",
                boxes(child, 9),
                "grid",
                compact(child)["grid9"],
                flush=True,
            )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
