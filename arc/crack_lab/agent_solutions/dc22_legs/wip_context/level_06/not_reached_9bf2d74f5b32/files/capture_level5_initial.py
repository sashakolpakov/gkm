"""Render the observed level-5 entry frame for visual mechanic comparison."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np
from PIL import Image

import players


PALETTE = np.asarray([
    (0, 0, 0), (0, 80, 220), (220, 35, 35), (35, 180, 55),
    (245, 205, 35), (130, 130, 130), (220, 45, 190),
    (245, 135, 25), (50, 180, 235), (125, 25, 55),
    (145, 220, 70), (40, 220, 210), (130, 75, 45),
    (245, 120, 180), (70, 70, 70), (245, 245, 245),
], dtype=np.uint8)


def observe(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    pixels = PALETTE[np.asarray(env.frame(), dtype=np.uint8)]
    image = Image.fromarray(pixels).resize(
        (512, 512), Image.Resampling.NEAREST
    )
    image.save("level5_initial.png")


arena.run_program("dc22", observe)
