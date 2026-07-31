"""Compact clean-room probes for lp85 level 4."""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from solve import solve
from perception import color_counts, connected_components, frame_delta


def reach_level_4(env):
    solve(env)


def compact_blobs(frame):
    return [
        (b.color, b.bbox, b.area, tuple(round(v, 1) for v in b.centroid))
        for b in connected_components(frame, min_area=3)
        if b.area < 1500
    ]


def candidate_points(frame):
    points = set()
    for b in connected_components(frame, min_area=3):
        if b.area >= 1500:
            continue
        r0, c0, r1, c1 = b.bbox
        points.update({
            ((c0 + c1) // 2, (r0 + r1) // 2),
            (int(round(b.centroid[1])), int(round(b.centroid[0]))),
            (c0, r0),
            (c1, r1),
        })
    return sorted(points)


CENTERS = (15, 45)
OFFSETS = ((-6, 0), (-3, 0), (0, -6), (0, -3), (0, 0),
           (0, 3), (0, 6), (3, 0), (6, 0))
HANDLES = (
    (15, 6), (15, 25), (15, 36), (15, 55),
    (45, 6), (45, 25), (45, 36), (45, 55),
    (6, 15), (25, 15), (36, 15), (55, 15),
    (6, 45), (25, 45), (36, 45), (55, 45),
)


def token_state(frame):
    f = np.asarray(frame)
    return tuple(
        tuple(int(f[r + dr, c + dc]) for dr, dc in OFFSETS)
        for r in CENTERS for c in CENTERS
    )


def run(env):
    reach_level_4(env)
    base = np.asarray(env.frame()).copy()
    Image.fromarray(np.uint8(base * 16)).resize(
        (512, 512), resample=Image.Resampling.NEAREST
    ).save("/tmp/lp85_l4.png")
    print("state", env.levels_completed, "terminal", env.terminal(),
          "actions", env.actions)
    print("counts", color_counts(base))
    print("blobs", compact_blobs(base))

    changed = []
    for x, y in candidate_points(base):
        clone = env.clone()
        before_level = clone.levels_completed
        clone.step(6, x, y)
        delta = frame_delta(base, clone.frame())
        if delta["count"] or clone.levels_completed != before_level:
            changed.append((x, y, delta["count"], delta["bbox"],
                            clone.levels_completed, clone.terminal()))
    print("responsive", changed)
    print("tokens0", token_state(base))
    unique = {}
    for x, y in HANDLES:
        clone = env.clone()
        clone.step(6, x, y)
        state = token_state(clone.frame())
        unique.setdefault(state, []).append((x, y))
    for i, (state, handles) in enumerate(unique.items()):
        print("operation", i, handles, state)


if __name__ == "__main__":
    A.run_program("lp85", run)
