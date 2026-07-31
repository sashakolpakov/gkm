"""Capture freshly reproduced sb26 layouts at each solved level."""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

import players


taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"tainted workspace: {taint_reason}")


def probe(env):
    for level in range(1, 6):
        plt.imsave(
            f"observed_level_{level}.png",
            env.frame(),
            vmin=0,
            vmax=15,
            cmap="tab20",
        )
        if level == 5:
            return
        getattr(players, f"play_level_{level}")(env)


levels, path, err = A.run_program("sb26", probe)
print(levels, len(path), err)
