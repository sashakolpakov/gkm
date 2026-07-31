"""Compactly reproduce the verified reward transitions in the saved prefix."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import color_counts, connected_components


def compact(frame):
    return {
        "colors": color_counts(frame),
        "actors": [
            (blob.color, blob.bbox, blob.area)
            for blob in connected_components(
                frame, colors=(7, 8, 9, 11, 12, 14, 15), min_area=3
            )
            if blob.bbox[0] < 63
        ],
    }


def probe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    previous = np.asarray(env.frame()).copy()
    old_level = int(env.levels_completed)
    for index, action in enumerate(path, 1):
        env.step(action)
        new_level = int(env.levels_completed)
        if new_level > old_level:
            print(
                "REWARD",
                {
                    "index": index,
                    "from": old_level + 1,
                    "action": action,
                    "before": compact(previous),
                    "after_level": new_level + 1,
                },
            )
            old_level = new_level
        previous = np.asarray(env.frame()).copy()


arena.run_program("bp35", probe)
