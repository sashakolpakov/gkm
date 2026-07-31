"""Compare staged shaft endpoints against their exact action costs."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact
from probe_l9_route_deletions import enter_level_9
from probe_l9_second_lower_entry import goals, staged_entry


def avatar(env):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9,), min_area=3)
        if blob.bbox[0] < 63
    )


def yellows(env):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(14,), min_area=3)
        if blob.bbox[0] < 63
    )


def probe(env):
    enter_level_9(env)
    for col in range(2, 9):
        child = staged_entry(env, col)
        for depth in range(1, 7):
            child.step(6, 3 + 6 * col, 33)
            if depth in (4, 5, 6):
                print(
                    "LANE",
                    col,
                    "depth",
                    depth,
                    "suffix_cost",
                    col + 3 + depth,
                    "terminal",
                    bool(child.terminal()),
                    "level",
                    int(child.levels_completed) + 1,
                    "avatar",
                    avatar(child),
                    "goals",
                    goals(child),
                    "yellow",
                    yellows(child),
                    "grid",
                    compact(child)["grid9"],
                    flush=True,
                )
            if child.terminal() or int(child.levels_completed) >= 9:
                break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
