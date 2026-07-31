"""Stop the first gap bridge early and let the gravity flip free-fall."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_control_row import compact, controls
from probe_l9_fast_prefix import enter_outer_controls_fast


def probe(env):
    root = env.clone()
    enter_outer_controls_fast(root)
    xs = (51, 45, 39, 33, 27, 21, 15, 9)
    print("CHAMBER", compact(root))
    for length in range(4, 9):
        child = root.clone()
        for x in xs[:length]:
            child.step(6, x, 45)
        staged = compact(child)
        child.step(*controls(child)[0])
        print(
            "BRIDGE",
            length,
            "staged",
            staged,
            "flipped",
            compact(child),
            "terminal",
            bool(child.terminal()),
            "controls",
            controls(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
