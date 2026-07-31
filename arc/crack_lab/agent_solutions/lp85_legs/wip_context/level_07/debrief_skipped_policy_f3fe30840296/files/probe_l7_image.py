"""Save the documented level-7 frame for direct visual inspection."""
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from solve import solve


def run(env):
    solve(env)
    palette = np.array([
        [0, 0, 0], [0, 80, 255], [255, 40, 40], [0, 170, 70],
        [255, 220, 0], [130, 130, 130], [230, 0, 190], [255, 130, 0],
        [0, 210, 230], [120, 0, 40], [80, 170, 255], [120, 255, 120],
        [150, 80, 255], [130, 75, 35], [245, 245, 245], [40, 40, 40],
    ], dtype=np.uint8)
    plt.imsave("/tmp/lp85_l7.png", palette[np.asarray(env.frame())])
    print("saved", env.levels_completed, env.actions)


if __name__ == "__main__":
    A.run_program("lp85", run)
