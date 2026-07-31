"""Open the yellow trapdoors before consuming the final retained switch."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_supported_final_alignment import aligned


def yellows(env):
    return tuple(
        (6, round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in connected_components(env.frame(), colors=(14,), min_area=3)
        if blob.bbox[0] < 63
    )


def run(root, name, plan):
    child = aligned(root, 5, 5)
    for token in plan:
        if token == "right_yellow":
            action = max(yellows(child), key=lambda item: item[1])
        elif token == "left_yellow":
            action = min(yellows(child), key=lambda item: item[1])
        else:
            action = token
        child.step(*action)
        if child.terminal():
            report((name, "PRE_TERMINAL", action), child)
            return
    report((name, "OPENED"), child)
    visible = controls(child)
    if not visible:
        return
    child.step(*visible[-1])
    report((name, "FLIP"), child)


def probe(env):
    enter_level_9(env)
    variants = {
        "RIGHT": ("right_yellow",),
        "LEFT": ("left_yellow",),
        "BOTH_RL": ("right_yellow", "left_yellow"),
        "BOTH_LR": ("left_yellow", "right_yellow"),
        "CATCH_BOTH": ((6, 21, 27), "right_yellow", "left_yellow"),
        "BOTH_CATCH": ("right_yellow", "left_yellow", (6, 21, 27)),
    }
    for name, plan in variants.items():
        run(env, name, plan)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
