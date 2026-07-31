"""Render level 7 immediately after its confirmed first checkpoint."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import numpy as np
from PIL import Image

import solve
from legs import (
    click_largest_color_9_submit_disc,
    learn_direction_protocol_from_selector,
)
from perception import arr, connected_components


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


def panel_geometry(frame):
    segments = [
        blob
        for blob in connected_components(frame, colors=(1, 5), min_area=3)
        if blob.area == 3
        and 1 in blob.size
        and blob.centroid[0] > len(frame) / 2
        and blob.centroid[1] > len(frame[0]) / 2
    ]
    return (
        sorted({int(round(blob.centroid[0])) for blob in segments}),
        sorted({int(round(blob.centroid[1])) for blob in segments}),
    )


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    codes = learn_direction_protocol_from_selector(env)
    rows, cols = panel_geometry(env.frame())
    frame = arr(env.frame()).copy()
    for col, symbol in zip(cols, "UURRRD"):
        for row_index, row in enumerate(rows):
            if (int(frame[row][col]) == 5) != (row_index in codes[symbol]):
                env.step(6, col, row)
    click_largest_color_9_submit_disc(env)

    pixels = PALETTE[np.asarray(env.frame(), dtype=np.uint8)]
    image = Image.fromarray(pixels).resize((768, 768), Image.Resampling.NEAREST)
    image.save("level_7_stage1.png")


levels, path, error = A.run_program("tn36", observe)
print("capture_result", {"levels": levels, "moves": len(path), "error": error})
