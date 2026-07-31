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


def selected_key(env):
    frame = np.asarray(env.frame())
    return np.where(np.isin(frame, (4, 9)), 0, frame).tobytes()


def probe(env):
    while env.levels_completed < 5:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    work = env.clone()
    for action in PROGRESS:
        work.step(*action) if isinstance(action, tuple) else work.step(action)
    work.step(6, 6, 46)
    path = bounded_bfs(
        work,
        lambda node, _: node.levels_completed >= 6,
        key_fn=selected_key,
        max_states=20000,
        max_depth=80,
    )
    print("FINAL_PATH", path)


arena.run_program("cn04", probe)
