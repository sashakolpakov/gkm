"""Decode the lower-left protocol selector on pristine level 6."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import PROTOCOL_ROWS
from perception import arr, connected_components, frame_delta


LEFT_ROWS = (33, 36, 39, 42, 45, 48)
LEFT_COLS = (11, 16, 21)


def glyphs(frame):
    inverse = {tuple(rows): symbol for symbol, rows in PROTOCOL_ROWS.items()}
    return "".join(
        inverse.get(
            tuple(index for index, row in enumerate(LEFT_ROWS) if int(frame[row][col]) == 5),
            "?",
        )
        for col in LEFT_COLS
    )


def yellow_mask(frame):
    blobs = [
        blob
        for blob in connected_components(frame, colors=(4,), min_area=4)
        if blob.bbox[2] < 32 and blob.bbox[1] < 32
    ]
    if not blobs:
        return None
    blob = max(blobs, key=lambda item: item.area)
    r0, c0, r1, c1 = blob.bbox
    return {
        "bbox": blob.bbox,
        "mask": tuple(
            "".join("#" if int(frame[row][col]) == 4 else "." for col in range(c0, c1 + 1))
            for row in range(r0, r1 + 1)
        ),
    }


def selected(frame):
    return [
        index
        for index, col in enumerate((5, 15, 25, 35))
        if int(frame[54][col]) == 9
    ]


def summary(env):
    return {
        "glyphs": glyphs(env.frame()),
        "selected": selected(env.frame()),
        "yellow": yellow_mask(env.frame()),
    }


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    print("initial", summary(env))
    for index, col in enumerate((5, 15, 25, 35)):
        clone = env.clone()
        before = arr(clone.frame()).copy()
        clone.step(6, col, 58)
        print(
            "select",
            index,
            {
                "summary": summary(clone),
                "delta": frame_delta(before, clone.frame()),
                "level_delta": clone.levels_completed - env.levels_completed,
            },
        )

    clone = env.clone()
    for index, col in enumerate((15, 25, 35, 5, 25)):
        before = arr(clone.frame()).copy()
        clone.step(6, col, 58)
        print(
            "sequence",
            index,
            {
                "click": col,
                "summary": summary(clone),
                "pixels": frame_delta(before, clone.frame())["count"],
            },
        )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
