"""Trace and continue the first verified level-5 eight transfer."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception as p
import players


PREFIX_RUNS = (
    (4, 5),
    (1, 3),
    (2, 1),
    (3, 2),
    (2, 1),
    (6, 1),
    (1, 6),
    (2, 1),
    (4, 3),
    (6, 1),
    (1, 2),
    (4, 6),
    (2, 2),
)


def state(env):
    data = p.arr(env.frame())
    avatar = int(np.argwhere(data[:53, :11] == 6)[:, 0].min())
    pieces = tuple(
        (blob.color, blob.bbox[0], blob.bbox[1])
        for blob in p.connected_components(
            env.frame(), colors=(8, 9), min_area=4
        )
        if blob.bbox[0] < 53
    )
    return avatar, pieces


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    print("START", state(env))
    for action, count in PREFIX_RUNS:
        for _ in range(count):
            env.step(action)
        print("RUN", action, count, state(env), env.levels_completed)
    print("DELTAS", {
        action: (delta["count"], delta["bbox"])
        for action, delta in p.action_deltas(env, env.actions).items()
    })


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
