import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import numpy as np
import gkm_arena as arena

import players
from perception import bounded_bfs


PROGRESS = [
    2, 2, 5,
    (6, 24, 26), 2, 3, 3, 3, 3, 3, 5,
    (6, 21, 21), 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4,
    (6, 56, 9), 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
]


def active_summary(frame):
    array = np.asarray(frame)
    mask = ~np.isin(array, (0, 4, 9))
    if not mask.any():
        return None
    rr, cc = np.where(mask)
    colors, counts = np.unique(array[mask], return_counts=True)
    return (
        int(mask.sum()),
        (int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())),
        tuple(zip(map(int, colors), map(int, counts))),
    )


def active_key(env):
    array = np.asarray(env.frame())
    return np.where(np.isin(array, (4, 9)), 0, array).tobytes()


def probe(env):
    while env.levels_completed < 5:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    work = env.clone()
    for action in PROGRESS:
        work.step(*action) if isinstance(action, tuple) else work.step(action)
    roots = {np.asarray(work.frame()).tobytes(): (None, work)}
    frame = np.asarray(work.frame())
    for row in range(1, 64, 3):
        for col in range(1, 64, 3):
            if int(frame[row, col]) in (0, 9):
                continue
            child = work.clone()
            child.step(6, col, row)
            key = np.asarray(child.frame()).tobytes()
            roots.setdefault(key, ((col, row), child))
    print("SELECTIONS", len(roots))
    for click, root in roots.values():
        print("TRY", click, active_summary(root.frame()), flush=True)
        path = bounded_bfs(
            root,
            lambda node, _: node.levels_completed >= 6,
            key_fn=active_key,
            max_states=8000,
            max_depth=48,
        )
        print(" PATH", path, flush=True)


arena.run_program("cn04", probe)
