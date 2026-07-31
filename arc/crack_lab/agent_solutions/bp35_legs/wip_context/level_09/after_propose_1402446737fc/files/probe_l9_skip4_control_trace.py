"""Locate the last usable controls along the four-skip prize-wall shortcut."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, route, step
from probe_l9_twelve_fast_frontier import SKIPS


def boxes(env, color):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(color,), min_area=2)
        if blob.bbox[0] < 63
    )


def probe(env):
    enter_level_9(env)
    child = env.clone()
    omitted = SKIPS | set(range(11, 15))
    previous = None
    applied = 0
    for index, (section, action) in enumerate(route()):
        if index in omitted:
            continue
        step(child, action)
        applied += 1
        signature = (
            controls(child),
            boxes(child, 7),
            boxes(child, 9),
            bool(child.terminal()),
        )
        if signature != previous:
            print(
                "EVENT",
                "index",
                index,
                "applied",
                applied,
                section,
                action,
                "controls",
                signature[0],
                "goals",
                signature[1],
                "avatar",
                signature[2],
                "terminal",
                signature[3],
                "grid",
                compact(child)["grid9"],
                flush=True,
            )
            previous = signature
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
