"""Collision and synchronization probes for the moving level-5 object."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import color_counts, connected_components, frame_delta
from probe_level5 import reach_level_5


TO_COLLISION = (1, 3, 3, 3, 2, 3, 3, 1, 3)


def summary(env):
    frame = np.asarray(env.frame())
    portrait = frame[53:63, 1:11]
    hud = max(
        (8, 9, 12, 14),
        key=lambda color: int(np.count_nonzero(portrait == color)),
    )
    zeros = tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(0,), min_area=1)
        if blob.bbox[0] < 55 and blob.bbox[1] >= 4
    )
    nines = tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(9,), min_area=1)
        if blob.bbox[0] < 55 and blob.bbox[1] >= 4
    )
    return {
        "level": int(env.levels_completed),
        "hud": hud,
        "zeros": zeros,
        "nines": nines,
        "counts": tuple(
            (color, color_counts(frame).get(color, 0))
            for color in (0, 1, 8, 9, 12, 14)
        ),
    }


def inspect(env):
    reach_level_5(env)
    clone = env.clone()
    for index, action in enumerate(TO_COLLISION, 1):
        before = np.asarray(clone.frame()).copy()
        clone.step(action)
        print(index, action, summary(clone), "delta", frame_delta(before, clone.frame())["count"])
    for action in env.actions:
        child = clone.clone()
        before = np.asarray(child.frame()).copy()
        child.step(action)
        print(
            "after_collision",
            action,
            summary(child),
            frame_delta(before, child.frame()),
        )


if __name__ == "__main__":
    levels, path, error = A.run_program("ls20", inspect)
    print("probe_result", levels, len(path), error)
