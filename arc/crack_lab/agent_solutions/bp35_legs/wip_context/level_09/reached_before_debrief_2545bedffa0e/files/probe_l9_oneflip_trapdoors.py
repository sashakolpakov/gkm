"""Open the aligned yellow trapdoor before the three-skip final flip."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_oneflip_lower import oneflip
from probe_l9_route_deletions import enter_level_9


def yellows(env):
    return tuple(
        (6, round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in connected_components(env.frame(), colors=(14,), min_area=3)
        if blob.bbox[0] < 63
    )


def run(root, lefts, mode):
    child = oneflip(root, lefts)
    visible = sorted(yellows(child), key=lambda action: action[1])
    if mode == "left":
        selected = visible[:1]
    elif mode == "right":
        selected = visible[-1:]
    else:
        selected = visible
    for action in selected:
        child.step(*action)
    report((lefts, mode, "OPENED"), child)
    switches = controls(child)
    if not switches or child.terminal():
        return
    child.step(*switches[-1])
    report((lefts, mode, "FLIP"), child)


def probe(env):
    enter_level_9(env)
    for lefts in range(3):
        for mode in ("left", "right", "both"):
            run(env, lefts, mode)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
