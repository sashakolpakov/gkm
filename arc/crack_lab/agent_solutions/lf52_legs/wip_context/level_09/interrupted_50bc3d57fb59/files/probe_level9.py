"""Compact clean-room observations at the checkpoint's current level."""

import json
import os
import sys

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from perception import action_deltas, color_counts, connected_components


def probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(action)

    frame = env.frame()
    blobs = [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=2)
    ]
    deltas = {
        action: (delta["count"], delta["bbox"])
        for action, delta in action_deltas(env).items()
    }
    print("STATE", env.levels_completed, tuple(env.actions), color_counts(frame))
    print("BLOBS", blobs)
    print("DELTAS", deltas)


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)
