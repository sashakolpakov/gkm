"""Targeted contact probes for level-5 objects."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import color_counts, connected_components, frame_delta
from probe_level5 import reach_level_5, world_marker_boxes


PATHS = {
    "target_blocked": (3, 2, 2, 1),
    "near_refill": (1, 3, 1, 1, 3, 4),
    "central_contact": (1, 3, 1, 1, 3, 3, 3),
    "lower_refill": (1, 3, 3, 3, 2, 3, 3, 2, 3, 3),
    "left_refill": (1, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 3, 3, 3),
}


def compact_state(env):
    frame = np.asarray(env.frame())
    color9 = [
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(9,), min_area=1)
        if blob.bbox[0] < 60 and blob.bbox[1] >= 4
    ]
    color12 = [
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(12,), min_area=1)
        if blob.bbox[0] < 60 and blob.bbox[1] >= 4
    ]
    refills = [
        blob.bbox
        for blob in connected_components(frame, colors=(11,), min_area=4)
        if blob.bbox[0] < 60
    ]
    return {
        "level": int(env.levels_completed),
        "energy": int(np.count_nonzero(frame[60:, :] == 11)),
        "color9": color9,
        "color12": color12,
        "refills": refills,
    }


def inspect(env):
    reach_level_5(env)
    root = env.clone()
    for name, path in PATHS.items():
        clone = root.clone()
        before = np.asarray(clone.frame()).copy()
        trace = []
        for action in path:
            if clone.terminal():
                break
            before = np.asarray(clone.frame()).copy()
            clone.step(action)
            trace.append((action, compact_state(clone)))
        print(name, "".join(map(str, path)))
        print("trace", trace)
        print("final", compact_state(clone))
        print("last_delta", frame_delta(before, clone.frame()))
        print("markers", world_marker_boxes(clone.frame()))
        print("colors", color_counts(clone.frame()))


if __name__ == "__main__":
    levels, path, error = A.run_program("ls20", inspect)
    print("probe_result", levels, len(path), error)
