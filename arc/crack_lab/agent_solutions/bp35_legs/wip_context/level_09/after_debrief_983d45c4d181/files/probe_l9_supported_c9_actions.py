"""Probe the local affordances at the far-right end of the lower gap corridor."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report
from probe_l9_supported_gap_climb import flipped


def c9(root):
    child = flipped(root, 6)
    for action in (
        (6, 45, 39),
        4,
        (6, 51, 39),
        4,
        (6, 57, 39),
        4,
    ):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def objects(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 9, 11, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63
        and (blob.color != 15 or blob.area == 21)
    )


def probe(env):
    enter_level_9(env)
    root = c9(env)
    report("ROOT", root)
    print("OBJECTS", objects(root), flush=True)
    actions = (
        3,
        4,
        7,
        (6, 57, 27),
        (6, 57, 33),
        (6, 57, 45),
        (6, 57, 51),
        (6, 51, 33),
        (6, 51, 45),
    )
    for action in actions:
        child = root.clone()
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report(("ACTION", action), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
