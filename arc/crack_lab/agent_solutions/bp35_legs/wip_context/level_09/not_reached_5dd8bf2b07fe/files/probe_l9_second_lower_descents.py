"""Descend the persistent shafts created by one lower pre-flip support."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_second_lower_entry import goals, staged_entry


def avatars(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(9,), min_area=3)
        if blob.bbox[0] < 63
    )


def run(root, col, verbose=False):
    child = staged_entry(root, col)
    if verbose:
        print("START", col, compact(child), "goals", goals(child))
    last = compact(child)
    for depth in range(1, 18):
        child.step(6, 3 + 6 * col, 33)
        last = compact(child)
        if verbose or goals(child) or int(child.levels_completed) >= 9:
            print(
                "DROP",
                col,
                depth,
                "terminal",
                bool(child.terminal()),
                "level",
                int(child.levels_completed) + 1,
                "controls",
                controls(child),
                "goals",
                goals(child),
                "avatar",
                avatars(child),
                "state",
                last,
            )
        if child.terminal() or int(child.levels_completed) >= 9:
            break
    if not verbose:
        print(
            "END",
            col,
            "terminal",
            bool(child.terminal()),
            "level",
            int(child.levels_completed) + 1,
            "goals",
            goals(child),
            "avatar",
            avatars(child),
            "state",
            last,
        )


def probe(env):
    enter_level_9(env)
    run(env, 2, verbose=True)
    for col in range(3, 9):
        run(env, col)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
