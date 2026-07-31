"""Compact clean-room observations for lp85 level 6."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import color_counts, connected_components, frame_delta
from solve import solve

CONTROLS = (
    (15, 28), (27, 15), (30, 58), (42, 45),
    (45, 28), (53, 55), (57, 15),
)


def compact_blobs(frame):
    return [
        (b.color, b.bbox, b.area, tuple(round(v, 1) for v in b.centroid))
        for b in connected_components(frame, min_area=2)
        if b.area < 2000
    ]


def candidate_points(frame):
    points = set()
    for b in connected_components(frame, min_area=2):
        if b.area >= 2000:
            continue
        r0, c0, r1, c1 = b.bbox
        points.add(((c0 + c1) // 2, (r0 + r1) // 2))
        points.add((int(round(b.centroid[1])), int(round(b.centroid[0]))))
    return sorted(points)


def token_state(frame):
    tokens = {}
    for b in connected_components(frame, min_area=4):
        if b.area == 4 and b.size == (2, 2) and b.bbox[0] >= 5:
            tokens[b.top_left] = b.color
    return tokens


def run(env):
    solve(env)
    base = np.asarray(env.frame()).copy()
    print("state", env.levels_completed, "terminal", env.terminal(),
          "actions", env.actions)
    print("counts", color_counts(base))
    print("blobs")
    for blob in compact_blobs(base):
        print(blob)
    print("responsive")
    for x, y in candidate_points(base):
        clone = env.clone()
        clone.step(6, x, y)
        delta = frame_delta(base, clone.frame())
        if delta["count"] or clone.levels_completed != env.levels_completed:
            print((x, y), delta["count"], delta["bbox"],
                  clone.levels_completed, clone.terminal())
    base_tokens = token_state(base)
    print("tokens", sorted(base_tokens.items()))
    for control in CONTROLS:
        clone = env.clone()
        clone.step(6, *control)
        after = token_state(clone.frame())
        changed = [(p, base_tokens.get(p), after.get(p))
                   for p in sorted(set(base_tokens) | set(after))
                   if base_tokens.get(p) != after.get(p)]
        print("control", control, "changed", changed)
    for control in CONTROLS:
        clone = env.clone()
        seen = {tuple(sorted(token_state(clone.frame()).items())): 0}
        stop = None
        for n in range(1, 31):
            clone.step(6, *control)
            key = tuple(sorted(token_state(clone.frame()).items()))
            if clone.levels_completed > 5 or clone.terminal() or key in seen:
                stop = (n, clone.levels_completed, clone.terminal(),
                        seen.get(key))
                break
            seen[key] = n
        print("repeat", control, stop)


if __name__ == "__main__":
    A.run_program("lp85", run)
