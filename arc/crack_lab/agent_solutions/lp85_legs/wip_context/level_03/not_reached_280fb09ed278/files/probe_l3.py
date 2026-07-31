"""Compact clean-room probes for lp85 level 3."""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import color_counts, connected_components, frame_delta


PREFIX = [
    (6, 5, 32), (6, 5, 32), (6, 5, 32), (6, 5, 32), (6, 5, 32),
    (6, 39, 17), (6, 48, 35), (6, 39, 17), (6, 39, 17),
    (6, 39, 17), (6, 48, 35), (6, 48, 35), (6, 48, 35),
]


def summarize(frame):
    blobs = connected_components(frame, min_area=3)
    compact = [
        (b.color, b.bbox, b.area)
        for b in blobs
        if b.area < 1500
    ]
    return color_counts(frame), compact


def token_map(frame):
    f = np.asarray(frame)
    rows = (19, 22, 25, 28, 31, 34, 37)
    cols = (15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45)
    return tuple(
        "".join("." if int(f[r, c]) in (3, 4) else f"{int(f[r, c]):X}"
                for c in cols)
        for r in rows
    )


def run(env):
    for action in PREFIX:
        env.step(*action)
    base = np.asarray(env.frame()).copy()
    Image.fromarray(np.uint8(base * 16)).resize((512, 512),
                                                resample=Image.Resampling.NEAREST).save("/tmp/lp85_l3.png")
    counts, blobs = summarize(base)
    print("state", env.levels_completed, "actions", env.actions)
    print("counts", counts)
    print("blobs", blobs)

    # Probe centers and corners of every compact component on independent clones.
    points = set()
    for b in connected_components(base, min_area=3):
        if b.area >= 1500:
            continue
        r0, c0, r1, c1 = b.bbox
        points.update({
            ((c0 + c1) // 2, (r0 + r1) // 2),
            (c0, r0),
            (c1, r1),
        })
    changed = []
    for x, y in sorted(points):
        clone = env.clone()
        before_level = clone.levels_completed
        clone.step(6, x, y)
        delta = frame_delta(base, clone.frame())
        if delta["count"] or clone.levels_completed != before_level:
            changed.append((x, y, delta["count"], delta["bbox"],
                            clone.levels_completed))
    print("responsive", changed)

    left, right = (6, 23, 41), (6, 35, 41)
    sequences = {
        "L": [left],
        "R": [right],
        "LL": [left] * 2,
        "RR": [right] * 2,
        "LR": [left, right],
        "RL": [right, left],
        "LLLL": [left] * 4,
        "RRRR": [right] * 4,
        "LRLR": [left, right] * 2,
        "RLRL": [right, left] * 2,
    }
    print("map0", token_map(base))
    for name, actions in sequences.items():
        clone = env.clone()
        for action in actions:
            clone.step(*action)
        print("seq", name, "level", clone.levels_completed,
              "map", token_map(clone.frame()),
              "counts", color_counts(clone.frame()))


if __name__ == "__main__":
    A.run_program("lp85", run)
