"""Trace the visible-but-uncontrolled 898 continuation."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception as p
import players
from probe_level5_transfer_path import PREFIX_RUNS


SUFFIX_RUNS = (
    (2, 5),
    (3, 6),
    (4, 1),
    (3, 4),
    (1, 1),
    (2, 4),
    (6, 1),
    (3, 4),
    (4, 3),
    (6, 1),
    (4, 2),
    (6, 1),
    (1, 1),
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


def carried(env):
    base = state(env)[1]
    out = []
    for action in (1, 2):
        branch = p.replay(env, [action])
        after = state(branch)[1]
        if after != base:
            out.append((action, after))
    return tuple(out)


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    for action, count in PREFIX_RUNS:
        for _ in range(count):
            env.step(action)
    print("TRANSFER", state(env), carried(env))
    for action, count in SUFFIX_RUNS:
        for _ in range(count):
            env.step(action)
        print("RUN", action, count, state(env), carried(env), env.levels_completed)
    base = state(env)
    for action in (1, 2, 3, 4, 6):
        branch = env.clone()
        for count in range(1, 7):
            branch.step(action)
            current = state(branch)
            if current != base or branch.levels_completed > env.levels_completed:
                print(
                    "PROBE",
                    action,
                    count,
                    current,
                    branch.levels_completed,
                )


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
