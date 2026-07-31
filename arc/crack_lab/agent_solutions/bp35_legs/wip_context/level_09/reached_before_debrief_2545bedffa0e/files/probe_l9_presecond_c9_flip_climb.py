"""Flip upward in exterior c9, then climb above the sealed hazard band."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_handoff_clickmap import avatar, relevant_full
from probe_l9_presecond_c9_descent import enter_c9
from probe_l9_route_deletions import enter_level_9


def goals(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=2)
        if blob.bbox[0] < 63
    )


def probe(env):
    enter_level_9(env)
    child = enter_c9(env)
    child.step(*controls(child)[0])
    for height in range(13):
        report(("HEIGHT", height), child)
        print(
            "STATE",
            height,
            "terminal",
            bool(child.terminal()),
            "avatar",
            avatar(child),
            "controls",
            controls(child),
            "goals",
            goals(child),
            "full",
            relevant_full(child),
            flush=True,
        )
        if child.terminal() or int(child.levels_completed) >= 9:
            return
        child.step(6, 57, 33)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
