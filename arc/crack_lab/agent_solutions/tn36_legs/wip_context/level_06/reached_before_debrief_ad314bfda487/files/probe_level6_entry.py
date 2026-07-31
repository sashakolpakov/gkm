"""Compact clean-room observations at pristine level-6 entry."""

import json
import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from perception import arr, color_counts, connected_components, frame_delta, object_candidates


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
    solve.solve(env)

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
    print("objects", object_candidates(frame, min_area=2))

    candidates = set()
    for blob in connected_components(frame):
        row, col = blob.centroid
        candidates.add((int(round(col)), int(round(row))))
        candidates.add((blob.bbox[1], blob.bbox[0]))

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
