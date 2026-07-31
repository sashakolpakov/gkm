import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import numpy as np
import gkm_arena as arena
from scipy import ndimage

import players
from perception import frame_delta


PROGRESS = [
    2, 2, 5,
    (6, 24, 26), 2, 3, 3, 3, 3, 3, 5,
    (6, 21, 21), 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4,
    (6, 56, 9), 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
]


def components(frame):
    array = np.asarray(frame)
    background = int(np.bincount(array.ravel()).argmax())
    labels, count = ndimage.label((array != background) & (array != 0))
    out = []
    for label in range(1, count + 1):
        rr, cc = np.where(labels == label)
        colors, amounts = np.unique(array[rr, cc], return_counts=True)
        out.append((
            len(rr),
            (int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())),
            tuple(zip(map(int, colors), map(int, amounts))),
        ))
    return sorted(out, reverse=True)


def probe(env):
    while env.levels_completed < 5:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    work = env.clone()
    for action in PROGRESS:
        work.step(*action) if isinstance(action, tuple) else work.step(action)
    print("final_frontier", work.levels_completed, components(work.frame()))
    for click in ((6, 46), (35, 0)):
        selected = work.clone()
        before = selected.frame().copy()
        selected.step(6, *click)
        print("select", click, frame_delta(before, selected.frame())["count"],
              components(selected.frame()))
        for action in range(1, 6):
            child = selected.clone()
            prior = child.frame().copy()
            child.step(action)
            delta = frame_delta(prior, child.frame())
            print(" action", action, delta["count"], delta["bbox"],
                  "level", child.levels_completed,
                  "components", components(child.frame()))


arena.run_program("cn04", probe)
