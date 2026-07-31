"""Pre-stage one catch column before flipping the second control chamber."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_lane9_second_walk import enter_lane9_second_control


def avatar(env):
    return [
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]


def probe(env):
    enter_lane9_second_control(env)
    root = env.clone()
    print("CHAMBER", compact(root))
    for col in (None, 2, 3, 4, 5, 6, 7, 8, 9):
        child = root.clone()
        if col is not None:
            child.step(6, 3 + 6 * col, 39)
        staged = compact(child)
        child.step(*controls(child)[0])
        flipped = compact(child)
        safe = 0
        before_terminal = compact(child)
        for _ in range(10):
            before_terminal = compact(child)
            child.step(4)
            if child.terminal():
                break
            safe += 1
        print(
            "STAGE",
            col,
            "staged",
            staged,
            "flipped",
            flipped,
            "safe_right",
            safe,
            "before_terminal",
            before_terminal,
            "terminal",
            bool(child.terminal()),
            "avatar",
            avatar(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
