"""Summarize the visible frame immediately before and after each known reward."""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import color_counts, connected_components, frame_delta


PATHS = (
    ((6, 5, 32),) * 5,
    ((6, 39, 17),) + ((6, 48, 35),)
    + ((6, 39, 17),) * 3 + ((6, 48, 35),) * 3,
    ((6, 23, 41),) * 4 + ((6, 35, 41),) * 4
    + ((6, 23, 41),) * 6 + ((6, 35, 41),) * 2,
    ((6, 15, 25),) * 4 + ((6, 6, 15),) * 8,
)


def compact(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=3)
        if b.area < 500
    ]


def rows(frame):
    """Run-length encode non-background spans for a compact symbolic picture."""
    f = np.asarray(frame)
    background = int(np.bincount(f.ravel()).argmax())
    out = []
    for r in range(64):
        spans = []
        c = 0
        while c < 64:
            if int(f[r, c]) == background:
                c += 1
                continue
            value = int(f[r, c])
            start = c
            while c + 1 < 64 and int(f[r, c + 1]) == value:
                c += 1
            spans.append((start, c, value))
            c += 1
        if spans:
            out.append((r, tuple(spans)))
    return out


def run(env):
    for level, path in enumerate(PATHS, 1):
        for action in path[:-1]:
            env.step(*action)
        before = np.asarray(env.frame()).copy()
        Image.fromarray(np.uint8(before * 16)).resize(
            (512, 512), resample=Image.Resampling.NEAREST
        ).save(f"/tmp/lp85_level{level}_reward_before.png")
        base_level = env.levels_completed
        env.step(*path[-1])
        after = np.asarray(env.frame()).copy()
        print("LEVEL", level, "reward", base_level, "->", env.levels_completed)
        print("delta", frame_delta(before, after))
        print("counts-before", color_counts(before))
        print("components-before", compact(before))
        print("rows-before", rows(before))


if __name__ == "__main__":
    A.run_program("lp85", run)
