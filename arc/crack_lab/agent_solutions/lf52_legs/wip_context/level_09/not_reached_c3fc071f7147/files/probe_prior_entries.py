"""Compact campaign-boundary observations for known wrapped-carrier levels."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import color_counts, connected_components


def compact(frame):
    return {
        "colors": color_counts(frame),
        "pieces": tuple(
            (blob.color, blob.bbox, blob.size, blob.area)
            for blob in connected_components(
                frame, colors=(1, 3, 7, 8, 9, 11, 12, 14, 15)
            )
            if (
                blob.color != 1
                or (blob.size == (4, 4) and blob.area == 16)
            )
        ),
    }


def probe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    prior = int(env.levels_completed)
    print("entry", prior + 1, 0, compact(env.frame()))
    for index, action in enumerate(path, 1):
        env.step(action)
        current = int(env.levels_completed)
        if current > prior:
            print("boundary", current, index)
            if current in (4, 5, 6, 7):
                print("entry", current + 1, index, compact(env.frame()))
            prior = current


arena.run_program("lf52", probe)
