"""Compact clean-room observations for lp85 level 7."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import color_counts, connected_components, frame_delta
from solve import solve


def compact_blobs(frame):
    return [
        (b.color, b.bbox, b.area, tuple(round(v, 1) for v in b.centroid))
        for b in connected_components(frame, min_area=2)
        if b.area < 2000
    ]


def run(env):
    solve(env)
    base = np.asarray(env.frame()).copy()
    print("state", env.levels_completed, "terminal", env.terminal(),
          "actions", env.actions)
    print("counts", color_counts(base))
    print("blobs")
    for blob in compact_blobs(base):
        print(blob)

    responsive = []
    for y in range(2, 64, 4):
        for x in range(2, 64, 4):
            clone = env.clone()
            clone.step(6, x, y)
            delta = frame_delta(base, clone.frame())
            if delta["count"] or clone.levels_completed != env.levels_completed:
                responsive.append(
                    ((x, y), delta["count"], delta["bbox"],
                     clone.levels_completed, clone.terminal())
                )
    print("responsive_grid", responsive)

    controls = ((22, 34), (32, 42))
    for control in controls:
        clone = env.clone()
        print("control", control)
        seen = {}
        for n in range(13):
            frame = np.asarray(clone.frame())
            key = frame[14:50, 8:56].tobytes()
            small = [
                (b.color, b.bbox, b.area)
                for b in connected_components(frame, min_area=2)
                if 19 <= b.bbox[0] <= 44 and 19 <= b.bbox[1] <= 44
                and b.area <= 12
            ]
            print(" step", n, "level", clone.levels_completed,
                  "seen", seen.get(key), "small", small)
            if clone.levels_completed != env.levels_completed or key in seen:
                break
            seen[key] = n
            clone.step(6, *control)

    chars = {1: "1", 2: "2", 3: ".", 4: "#", 5: "5", 8: "8",
             9: "9", 10: "A", 11: "B", 14: "E", 15: "F"}
    frame = np.asarray(env.frame())
    print("crop")
    for r in range(14, 46):
        print(f"{r:02}", "".join(chars[int(v)] for v in frame[r, 14:50]))


if __name__ == "__main__":
    A.run_program("lp85", run)
