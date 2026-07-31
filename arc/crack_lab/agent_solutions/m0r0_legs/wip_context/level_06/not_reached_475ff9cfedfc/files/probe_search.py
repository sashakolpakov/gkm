import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
import players


def symbolic_key(env):
    frame = P.arr(env.frame()).copy()
    frame[np.isin(frame, (0, 6, 7))] = 6
    return frame.tobytes()


def run(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
    starts = (
        (),
        ((6, 31, 43), 4, (6, 19, 23)),
        ((6, 31, 43), 4, 4, 4, 4, 4, (6, 19, 23)),
    )
    for start in starts:
        staged = env.clone()
        for action in start:
            if isinstance(action, tuple):
                staged.step(*action)
            else:
                staged.step(action)
        path = P.bounded_bfs(
            staged,
            P.level_goal(5),
            actions=(1, 2, 3, 4, 5),
            key_fn=symbolic_key,
            max_states=30000,
            max_depth=120,
        )
        print("SEARCH", start, path)


A.run_program("m0r0", run)
