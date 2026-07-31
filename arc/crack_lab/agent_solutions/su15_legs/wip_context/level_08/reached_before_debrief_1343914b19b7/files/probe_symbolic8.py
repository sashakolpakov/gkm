import json

import gkm_try as H

from perception import color_counts, connected_components
from probe_finish8 import PREFIX


def summary(env, ring_mask):
    frame = env.frame()
    squares = tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=4)
        if (
            blob.bbox[0] >= 10
            and blob.size[0] == blob.size[1]
            and blob.area == blob.size[0] ** 2
            and blob.color not in (3, 4, 5, 7, 9)
        )
    )
    body_pixels = {
        (row, col)
        for row in range(10, 64)
        for col in range(64)
        if int(frame[row][col]) == 7
    }
    counts = color_counts(frame)
    return {
        "level": int(env.levels_completed),
        "terminal": bool(env.terminal()),
        "squares": squares,
        "rings": counts.get(9, 0),
        "body_overlap": len(body_pixels & ring_mask),
    }


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    initial = env.frame()
    ring_mask = {
        (row, col)
        for row in range(10, 64)
        for col in range(64)
        if int(initial[row][col]) == 9
    }
    components = tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(initial, min_area=2)
        if blob.bbox[0] >= 10 and blob.color not in (4, 5, 9)
    )
    print("ROOT", summary(env, ring_mask), "components", components)
    for index, action in enumerate(PREFIX, 1):
        env.step(*action)
        print(index, action, summary(env, ring_mask))


H.A.run_program("su15", inspect)
