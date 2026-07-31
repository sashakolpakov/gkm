"""Render the reproduced pristine level-6 entry for local inspection."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import numpy as np
from PIL import Image

import solve


PALETTE = np.asarray(
    [
        (0, 0, 0),
        (0, 110, 255),
        (235, 45, 55),
        (30, 190, 70),
        (255, 220, 0),
        (145, 145, 145),
        (220, 55, 210),
        (255, 145, 30),
        (80, 220, 230),
        (125, 25, 45),
        (255, 245, 210),
        (85, 210, 255),
        (125, 70, 180),
        (75, 75, 75),
        (40, 80, 130),
        (245, 245, 245),
    ],
    dtype=np.uint8,
)


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)
    pixels = PALETTE[np.asarray(env.frame(), dtype=np.uint8)]
    image = Image.fromarray(pixels).resize((768, 768), Image.Resampling.NEAREST)
    image.save("level_6_entry.png")


levels, path, error = A.run_program("tn36", observe)
print("capture_result", {"levels": levels, "moves": len(path), "error": error})
