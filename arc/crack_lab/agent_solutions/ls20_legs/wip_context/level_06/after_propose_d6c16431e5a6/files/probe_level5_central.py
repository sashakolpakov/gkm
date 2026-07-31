"""Contextual probes for the four-colour central object."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from probe_level5 import reach_level_5


TO_CENTRAL = (1, 3, 1, 1, 3, 3, 3)


def hud(frame):
    crop = np.asarray(frame)[53:63, 1:11]
    return tuple(
        (color, int(np.count_nonzero(crop == color)))
        for color in (8, 9, 12, 14)
    )


def inspect(env):
    reach_level_5(env)
    root = env.clone()
    for suffix in (
        (),
        (1,),
        (2,),
        (3,),
        (4,),
        (4, 3),
        (4, 4),
        (4, 3, 4),
        (4, 3, 4, 3),
        (4, 3, 4, 3, 4, 3),
    ):
        clone = root.clone()
        trace = [hud(clone.frame())]
        for action in TO_CENTRAL + suffix:
            clone.step(action)
            trace.append(hud(clone.frame()))
        print("suffix", suffix, "hud_trace", trace[-4:])


if __name__ == "__main__":
    levels, path, error = A.run_program("ls20", inspect)
    print("probe_result", levels, len(path), error)
