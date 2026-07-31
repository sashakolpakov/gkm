"""Compact clean-room observations for lp85 level 8."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import color_counts, connected_components, frame_delta
from solve import solve


def run(env):
    solve(env)
    base = np.asarray(env.frame()).copy()
    print("state", env.levels_completed, "terminal", env.terminal(),
          "actions", env.actions)
    print("counts", color_counts(base))
    print("blobs", [
        (b.color, b.bbox, b.area)
        for b in connected_components(base, min_area=2)
        if b.area < 1000
    ])

    responsive = []
    for y in range(2, 64, 4):
        for x in range(2, 64, 4):
            clone = env.clone()
            clone.step(6, x, y)
            delta = frame_delta(base, clone.frame())
            if delta["count"] or clone.levels_completed != env.levels_completed:
                responsive.append(
                    ((x, y), delta["count"], delta["bbox"],
                     clone.levels_completed)
                )
    print("responsive", responsive)

    # Persistent colored 2x2 tokens form the puzzle state.
    token_colors = {1, 2, 9, 10, 11, 15}

    def tokens(frame):
        return {
            (b.bbox[1], b.bbox[0]): b.color
            for b in connected_components(frame, colors=token_colors, min_area=4)
            if b.area == 4 and b.size == (2, 2) and b.bbox[1] < 48
        }

    initial = tokens(base)
    print("token_rows")
    for y in sorted({p[1] for p in initial}):
        print(y, sorted((x, initial[(x, y)]) for x, yy in initial if yy == y))

    controls = ((50, 24), (50, 29), (50, 34), (31, 57))
    for control in controls:
        clone = env.clone()
        print("control", control)
        seen = {}
        for n in range(16):
            state = tuple(sorted(tokens(clone.frame()).items()))
            if state in seen or clone.levels_completed != env.levels_completed:
                print(" stop", n, "repeat", seen.get(state),
                      "level", clone.levels_completed)
                break
            seen[state] = n
            if n == 0:
                before = dict(state)
                clone.step(6, *control)
                after = tokens(clone.frame())
                print(" changed", sorted(
                    (p, before.get(p), after.get(p))
                    for p in set(before) | set(after)
                    if before.get(p) != after.get(p)
                ))
            else:
                clone.step(6, *control)


if __name__ == "__main__":
    A.run_program("lp85", run)
