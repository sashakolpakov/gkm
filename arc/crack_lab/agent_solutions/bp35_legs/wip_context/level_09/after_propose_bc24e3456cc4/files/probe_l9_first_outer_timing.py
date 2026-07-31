"""Flip the first outer descent at the earliest visible control."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import compact, controls, enter_control_row


def probe(env):
    base = env.clone()
    for depth in range(6, 13):
        child = base.clone()
        enter_control_row(child)
        child.step(6, 9, 3)
        for _ in range(depth):
            child.step(6, 3, 33)
        visible = controls(child)
        before = compact(child)
        if visible:
            child.step(*visible[-1])
        print(
            "TIMING",
            depth,
            "visible",
            visible,
            "before",
            before,
            "after",
            compact(child),
            "terminal",
            bool(child.terminal()),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
