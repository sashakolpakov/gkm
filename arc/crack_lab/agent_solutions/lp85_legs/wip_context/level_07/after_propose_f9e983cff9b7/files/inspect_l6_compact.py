"""Compact level-6 visual/state probe using only the documented frame surface."""
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import frame_delta
from probe_l6 import CONTROLS
from solve import solve


def run(env):
    solve(env)
    frame = np.asarray(env.frame(), dtype=int)
    palette = np.array([
        [0, 0, 0], [0, 80, 255], [255, 40, 40], [0, 170, 70],
        [255, 220, 0], [130, 130, 130], [230, 0, 190], [255, 130, 0],
        [0, 210, 230], [120, 0, 40], [80, 170, 255], [120, 255, 120],
        [150, 80, 255], [130, 75, 35], [245, 245, 245], [40, 40, 40],
    ], dtype=np.uint8)
    plt.imsave("/tmp/lp85_l6.png", palette[frame])

    print("level", env.levels_completed, "actions", env.actions)
    for point in ((3, 3),) + CONTROLS:
        clone = env.clone()
        clone.step(6, *point)
        delta = frame_delta(frame, clone.frame())
        border = [s for s in delta["samples"] if s[0] <= 1 or s[1] <= 1]
        print(point, "changed", delta["count"], "border", border[:12])


if __name__ == "__main__":
    A.run_program("lp85", run)
