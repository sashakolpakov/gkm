"""Compact clean-room probes for lp85 level 5."""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import color_counts, connected_components, frame_delta
from solve import solve

SLOTS = (
    *((6, c) for c in (17, 23, 29, 35, 41)),
    (12, 17),
    *((18, c) for c in (17, 23, 29)),
    (24, 29),
    *((30, c) for c in (17, 23, 29)),
    (36, 17),
    *((42, c) for c in (17, 23, 29)),
    (48, 29),
    *((54, c) for c in (17, 23, 29)),
)
ACTIONS = ((6, 9, 7), (6, 51, 7), (6, 11, 37), (6, 37, 37))


def tile_state(frame):
    f = np.asarray(frame)
    return tuple(int(f[r, c]) for r, c in SLOTS)


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
        points.add(((c0 + c1) // 2, (r0 + r1) // 2))
        points.add((int(round(b.centroid[1])), int(round(b.centroid[0]))))
    return sorted(points)


def run(env):
    solve(env)
    base = np.asarray(env.frame()).copy()
    Image.fromarray(np.uint8(base * 16)).resize(
        (512, 512), resample=Image.Resampling.NEAREST
    ).save("level5_frame.png")
    print("state", env.levels_completed, "terminal", env.terminal(),
          "actions", env.actions)
    print("counts", color_counts(base))
    print("blobs", compact_blobs(base))
    print("tiles", tile_state(base))
    for action in ACTIONS:
        clone = env.clone()
        clone.step(*action)
        print("action", action[1:], "tiles", tile_state(clone.frame()),
              "level", clone.levels_completed)
    for action in ACTIONS:
        clone = env.clone()
        seen = {tile_state(clone.frame()): 0}
        result = []
        for n in range(1, 41):
            clone.step(*action)
            state = tile_state(clone.frame())
            if clone.levels_completed > 4 or clone.terminal() or state in seen:
                result = [n, clone.levels_completed, clone.terminal(),
                          seen.get(state)]
                break
            seen[state] = n
        print("repeat", action[1:], result)
    changed = []
    for x, y in candidate_points(base):
        clone = env.clone()
        before_level = clone.levels_completed
        clone.step(6, x, y)
        delta = frame_delta(base, clone.frame())
        if delta["count"] or clone.levels_completed != before_level:
            vals0, cnt0 = np.unique(base, return_counts=True)
            vals1, cnt1 = np.unique(np.asarray(clone.frame()), return_counts=True)
            changed.append((
                (x, y), delta["count"], delta["bbox"],
                dict(zip(map(int, vals0), map(int, cnt0))),
                dict(zip(map(int, vals1), map(int, cnt1))),
                clone.levels_completed, clone.terminal(),
            ))
    print("responsive")
    for item in changed:
        print(item)


if __name__ == "__main__":
    A.run_program("lp85", run)
