"""Compact clean-room observations at pristine level-7 entry."""

import json
import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from perception import arr, color_counts, connected_components, frame_delta


def transition_counts(before, after):
    changed = arr(before) != arr(after)
    return tuple(
        sorted(
            Counter(
                zip(
                    (int(value) for value in arr(before)[changed]),
                    (int(value) for value in arr(after)[changed]),
                )
            ).items()
        )
    )


def compact_blobs(frame):
    return [
        (blob.color, blob.bbox, blob.area, tuple(round(v, 1) for v in blob.centroid))
        for blob in connected_components(frame, min_area=2)
    ]


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    frame = arr(env.frame())
    half_r, half_c = frame.shape[0] // 2, frame.shape[1] // 2
    print(
        "entry",
        {
            "level": env.levels_completed + 1,
            "actions": env.actions,
            "shape": frame.shape,
            "colors": color_counts(frame),
            "quadrants": {
                "UL": color_counts(frame[:half_r, :half_c]),
                "UR": color_counts(frame[:half_r, half_c:]),
                "LL": color_counts(frame[half_r:, :half_c]),
                "LR": color_counts(frame[half_r:, half_c:]),
            },
        },
    )
    print("blobs", compact_blobs(frame))

    candidates = set()
    for blob in connected_components(frame, min_area=2):
        row, col = blob.centroid
        candidates.add((int(round(col)), int(round(row))))
        candidates.add((blob.bbox[1], blob.bbox[0]))
        candidates.add(
            (
                (blob.bbox[1] + blob.bbox[3]) // 2,
                (blob.bbox[0] + blob.bbox[2]) // 2,
            )
        )

    effects = []
    for col, row in sorted(candidates, key=lambda point: (point[1], point[0])):
        clone = env.clone()
        before = arr(clone.frame()).copy()
        clone.step(6, col, row)
        after = arr(clone.frame()).copy()
        delta = frame_delta(before, after)
        level_delta = clone.levels_completed - env.levels_completed
        if delta["count"] or level_delta:
            effects.append(
                {
                    "at": (col, row),
                    "level_delta": level_delta,
                    "pixels": delta["count"],
                    "bbox": delta["bbox"],
                    "transitions": transition_counts(before, after),
                }
            )
    print("click_effects", effects)


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
