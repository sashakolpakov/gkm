"""Descend the PRE_SECOND exterior c9 lane while retaining its last control."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_handoff_clickmap import avatar, handoff, relevant_full
from probe_l9_route_deletions import enter_level_9


def enter_c9(root):
    child = handoff(root)
    child.step(6, 57, 29)
    child.step(4)
    return child


def probe(env):
    enter_level_9(env)
    child = enter_c9(env)
    print(
        "C9_OBJECTS",
        tuple(
            (blob.color, blob.bbox, blob.area, blob.centroid)
            for blob in connected_components(
                child.frame(), colors=(12, 14, 15), min_area=1
            )
            if blob.bbox[0] < 63 and blob.bbox[3] >= 36
        ),
        flush=True,
    )
    for depth in range(13):
        report(("DEPTH", depth), child)
        print(
            "STATE",
            depth,
            "terminal",
            bool(child.terminal()),
            "avatar",
            avatar(child),
            "controls",
            controls(child),
            "full",
            relevant_full(child),
            flush=True,
        )
        if child.terminal() or int(child.levels_completed) >= 9:
            return
        child.step(6, 57, 35)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
