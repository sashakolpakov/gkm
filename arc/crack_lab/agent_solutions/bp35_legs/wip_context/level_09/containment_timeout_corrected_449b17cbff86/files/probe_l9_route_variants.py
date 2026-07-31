"""Compare minimal wall routes built from each safe first-shelf lane."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_gap_flip import enter_gap_landing


def avatar(env):
    return [
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]


def build_variant(aligned, lane):
    child = aligned.clone()
    for col in range(1, lane + 1):
        child.step(6, 3 + 6 * col, 27)
        child.step(4)
    x = 3 + 6 * lane
    for _ in range(6):
        child.step(6, x, 35)
    if lane <= 7:
        child.step(6, 45, 35)
    for col in (6, 7, 8):
        child.step(6, 3 + 6 * col, 41)
    for col in range(lane + 1, 10):
        child.step(6, 3 + 6 * col, 27)
        child.step(4)
    child.step(6, 57, 35)
    return child


def probe(env):
    enter_gap_landing(env)
    env.step(6, 3, 35)
    aligned = env.clone()
    for lane in (2, 3, 4, 5, 6, 7, 8):
        child = build_variant(aligned, lane)
        outside = compact(child)
        if not child.terminal():
            for _ in range(5):
                child.step(6, 57, 35)
        visible = controls(child)
        if visible and not child.terminal():
            child.step(*visible[0])
        after_flip = compact(child)
        safe_left = 0
        before_terminal = compact(child)
        for _ in range(9):
            before_terminal = compact(child)
            child.step(3)
            if child.terminal():
                break
            safe_left += 1
        print(
            "VARIANT",
            lane,
            "outside",
            outside,
            "after_flip",
            after_flip,
            "safe_left",
            safe_left,
            "before_terminal",
            before_terminal,
            "terminal",
            bool(child.terminal()),
            "avatar",
            avatar(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
