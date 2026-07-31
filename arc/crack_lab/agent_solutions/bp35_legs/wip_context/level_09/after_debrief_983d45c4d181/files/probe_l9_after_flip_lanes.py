"""Compare ceiling openings after the far-right gravity reversal."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_after_wall_flip import enter_after_wall_flip


def goals(env):
    return [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
        if blob.bbox[0] < 63
    ]


def avatar(env):
    return [
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]


def probe(env):
    base = env.clone()
    for col in range(9, 2, -1):
        child = base.clone()
        enter_after_wall_flip(child)
        for _ in range(9 - col):
            child.step(3)
            if child.terminal():
                break
        walked = (bool(child.terminal()), avatar(child), compact(child))
        if not child.terminal():
            child.step(6, 3 + 6 * col, 33)
        print(
            "LANE",
            col,
            "walked",
            walked,
            "opened_terminal",
            bool(child.terminal()),
            "avatar",
            avatar(child),
            "controls",
            controls(child),
            "goals",
            goals(child),
            "state",
            compact(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
