import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

A = (6, 56, 8)
B = (6, 51, 25)
S = (6, 51, 48)
D = {
    "u": (6, 50, 32),
    "d": (6, 50, 40),
    "l": (6, 46, 36),
    "r": (6, 54, 36),
}
REMOTE = (
    [3] * 5
    + [A] * 4
    + [2, 2, 3, 3, 3, 2, 3, A, 1, A, 1, 1, B]
    + [1] * 17
    + [3]
    + [2] * 11
)
ROOT_TO_SELECTOR = (
    [2] * 8
    + [4, 4, A, 4, A, 1]
    + [A, 4] * 3
    + [1, 1, 1]
)
HUB = REMOTE + ROOT_TO_SELECTOR + [S, S, S, 3, B]
REVERSE_TO_ROOT = (
    [4, 2, 2, 2, 3, 2, 3, 3, 2, 3, 3, 1, A, 1]
    + [1] * 7
    + [3]
)


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, actions):
    for action in actions:
        step(env, action)


def avatar(env):
    rows, cols = np.where(np.asarray(env.frame())[:62, :40] == 14)
    return None if not len(rows) else (int(rows.min() // 2), int(cols.min() // 2))


def ring(env):
    return [
        (blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(8,), min_area=4
        )
        if blob.bbox[1] < 40
    ]


def move_ring_remotely(env, labels):
    movement = {"u": (1, 2), "d": (2, 1), "l": (3, 4), "r": (4, 3)}
    for label in labels:
        outward, inward = movement[label]
        apply(env, [outward, D[label], inward])


def compact_trace(root, label, actions):
    env = root.clone()
    events = []
    previous = (avatar(env), ring(env))
    for index, action in enumerate(actions, 1):
        step(env, action)
        current = (avatar(env), ring(env))
        if current != previous or env.levels_completed > 5:
            events.append((index, action, current, env.levels_completed))
        previous = current
        if env.levels_completed > 5:
            break
    print(label, "events", events, flush=True)


def run(env):
    solver.solve(env)
    apply(env, HUB)
    move_ring_remotely(env, ["l", "l", "l"])
    apply(env, [B] + REVERSE_TO_ROOT + [1] * 5)
    apply(env, [B] + [2] * 5 + ROOT_TO_SELECTOR + [3, B])
    print("BOARDED", avatar(env), ring(env), env.levels_completed, flush=True)

    for label, control in D.items():
        compact_trace(env, f"DIRECT_{label}", [control] * 8)
    compact_trace(env, "DIRECT_ROUTE", [D["r"]] * 4 + [D["u"]] * 8)
    compact_trace(env, "REMOTE_ROUTE", sum(
        ([{"u": 1, "d": 2, "l": 3, "r": 4}[label],
          D[label],
          {"u": 2, "d": 1, "l": 4, "r": 3}[label]]
         for label in ("r", "r", "r", "u", "r", "u", "u", "l", "l", "u", "u", "u")),
        [],
    ))


arena.run_program("dc22", run)
