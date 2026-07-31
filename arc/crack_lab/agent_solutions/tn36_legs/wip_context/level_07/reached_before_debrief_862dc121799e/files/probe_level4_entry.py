"""Compact clean-room observations at pristine level-4 entry."""

import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

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


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    frame = arr(env.frame())
    print(
        "entry",
        {
            "level": env.levels_completed + 1,
            "actions": env.actions,
            "shape": frame.shape,
            "colors": color_counts(frame),
        },
    )

    for color in sorted(color_counts(frame)):
        blobs = connected_components(frame, colors=(color,))
        compact = [
            (blob.bbox, blob.area, blob.size, tuple(round(v, 1) for v in blob.centroid))
            for blob in blobs
            if blob.area < frame.size
        ]
        print("components", color, compact)

    candidates = set()
    for blob in connected_components(frame):
        row, col = blob.centroid
        candidates.add((int(round(col)), int(round(row))))
    for row in range(2, 64, 4):
        for col in range(2, 64, 4):
            candidates.add((col, row))

    effects = defaultdict(list)
    for col, row in sorted(candidates, key=lambda point: (point[1], point[0])):
        clone = env.clone()
        before = arr(clone.frame()).copy()
        clone.step(6, col, row)
        after = arr(clone.frame()).copy()
        delta = frame_delta(before, after)
        if delta["count"] or clone.levels_completed != env.levels_completed:
            key = (
                clone.levels_completed - env.levels_completed,
                delta["count"],
                delta["bbox"],
                transition_counts(before, after),
            )
            effects[key].append((col, row))

    for effect, clicks in effects.items():
        print("effect", effect, "clicks", clicks)


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
