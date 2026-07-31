"""Turn upward from each safely reachable lane after the early flip."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_early_flip_climb import enter_early_flip


def goals(env):
    return [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
        if blob.bbox[0] < 63
    ]


def probe(env):
    base = env.clone()
    for left_steps in range(4):
        child = base.clone()
        enter_early_flip(child, 0)
        for _ in range(left_steps):
            child.step(3)
        col = 9 - left_steps
        before = compact(child)
        child.step(6, 3 + 6 * col, 33)
        print(
            "TURN",
            "left",
            left_steps,
            "col",
            col,
            "before",
            before,
            "terminal",
            bool(child.terminal()),
            "controls",
            controls(child),
            "goals",
            goals(child),
            "after",
            compact(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
