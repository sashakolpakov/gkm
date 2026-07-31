"""Render selected observed frames with a fixed palette for visual inspection."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np
from PIL import Image

import solve
from probe_l6_cargo_top import CARGO_TOP_PATH
from probe_l6_right import enter_right


PALETTE = np.asarray([
    (0, 0, 0),
    (0, 80, 220),
    (220, 35, 35),
    (35, 180, 55),
    (245, 205, 35),
    (130, 130, 130),
    (220, 45, 190),
    (245, 135, 25),
    (50, 180, 235),
    (125, 25, 55),
    (145, 220, 70),
    (40, 220, 210),
    (130, 75, 45),
    (245, 120, 180),
    (70, 70, 70),
    (245, 245, 245),
], dtype=np.uint8)


def save(frame, path):
    pixels = PALETTE[np.asarray(frame, dtype=np.uint8)]
    Image.fromarray(pixels).resize((512, 512), Image.Resampling.NEAREST).save(path)


def observe(env):
    solve.solve(env)
    save(env.frame(), "level6_initial.png")
    lifted = enter_right(env, 3)
    lifted.step(1)
    lifted.step(6, 50, 34)
    save(lifted.frame(), "level6_lifted.png")
    cargo_top = enter_right(env, 3)
    for action in CARGO_TOP_PATH:
        if isinstance(action, tuple):
            cargo_top.step(*action)
        else:
            cargo_top.step(action)
    save(cargo_top.frame(), "level6_cargo_top.png")


arena.run_program("dc22", observe)
