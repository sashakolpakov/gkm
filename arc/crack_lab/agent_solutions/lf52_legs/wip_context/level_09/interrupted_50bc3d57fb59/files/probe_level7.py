"""Compact public observation of the validated level-7 entry."""

import json
import sys

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from perception import action_deltas, color_counts, connected_components


PREFIX_END = 331


def probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"][:PREFIX_END]:
        env.step(action)
    frame = env.frame()
    blobs = tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=2)
        if blob.color != 10
    )
    deltas = {
        action: (delta["count"], delta["bbox"])
        for action, delta in action_deltas(env).items()
    }
    print("LEVEL7_STATE", env.levels_completed, tuple(env.actions), color_counts(frame))
    print("LEVEL7_BLOBS", blobs)
    print("LEVEL7_DELTAS", deltas)


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)
