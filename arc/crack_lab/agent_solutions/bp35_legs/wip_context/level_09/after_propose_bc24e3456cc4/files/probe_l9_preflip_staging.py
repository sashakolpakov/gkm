"""Test catch staging before the depth-eight gravity reversal."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact
from probe_l9_wall_depth8 import enter_wall_depth8


def avatar(env):
    return [
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]


def full_catches(env):
    return [
        (round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in connected_components(env.frame(), colors=(15,), min_area=3)
        if blob.bbox[0] < 63 and blob.area == 21
    ]


def probe(env):
    sequences = (
        (),
        ((51, 29),),
        ((51, 35),),
        ((51, 29), (51, 35)),
        ((51, 35), (51, 29)),
        ((45, 29),),
        ((45, 35),),
        ((51, 29), (45, 29)),
        ((51, 35), (45, 35)),
    )
    base = env.clone()
    for sequence in sequences:
        child = base.clone()
        enter_wall_depth8(child)
        staged_ok = True
        for target in sequence:
            child.step(6, *target)
            if child.terminal():
                staged_ok = False
                break
        before_flip = compact(child)
        if staged_ok:
            child.step(6, 3, 41)
        after_flip = compact(child)
        catches = full_catches(child)
        if not child.terminal():
            child.step(3)
        print(
            "SEQUENCE",
            sequence,
            "staged_ok",
            staged_ok,
            "before",
            before_flip,
            "after_flip",
            after_flip,
            "catches",
            catches,
            "move_terminal",
            bool(child.terminal()),
            "avatar",
            avatar(child),
            "after_move",
            compact(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
