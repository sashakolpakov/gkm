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
    "u": (1, (6, 50, 32), 2),
    "d": (2, (6, 50, 40), 1),
    "l": (3, (6, 46, 36), 4),
    "r": (4, (6, 54, 36), 3),
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
TOP_PATH = ("u", "r", "u", "u", "l", "l", "u", "u", "u")


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, actions):
    for action in actions:
        step(env, action)


def avatar(env):
    rows, cols = np.where(np.asarray(env.frame())[:62, :40] == 14)
    return None if not len(rows) else (int(rows.min() // 2), int(cols.min() // 2))


def move_ring(env, labels):
    for label in labels:
        apply(env, D[label])


def components(env, color):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(color,), min_area=2
        )
        if blob.bbox[1] < 40
    )


def solid_goals(env):
    frame = np.asarray(env.frame())
    return tuple(
        (row, col)
        for row in range(31)
        for col in range(20)
        if np.all(frame[
            2 * row:2 * row + 2,
            2 * col:2 * col + 2,
        ] == 11)
    )


def return_to_hub(selector):
    matches = []
    for phase in range(4):
        child = selector.clone()
        apply(child, [S] * phase + [3, B])
        matches.append((phase, avatar(child), child))
    print(
        "HUB_CHOICES", [(phase, position) for phase, position, _ in matches],
        flush=True,
    )
    for phase, position, child in matches:
        if position == (29, 17):
            return phase, child
    return None, None


def run(env):
    solver.solve(env)
    apply(env, HUB)
    move_ring(env, ("l", "l", "l"))
    docked = env.clone()
    print(
        "DOCK", components(docked, 8), components(docked, 12),
        avatar(docked), flush=True,
    )

    for cycles in (0, 2, 4):
        cargo = docked.clone()
        apply(cargo, [B] + REVERSE_TO_ROOT)
        apply(cargo, [1, 1, 1, 4, 4] + [B] * cycles)
        apply(cargo, [3, 3, 2, 2, 2] + ROOT_TO_SELECTOR)
        print(
            "SELECTOR", cycles, avatar(cargo),
            components(cargo, 8), components(cargo, 12), flush=True,
        )
        phase, returned = return_to_hub(cargo)
        if returned is None:
            continue
        before = (components(returned, 8), components(returned, 12))
        move_ring(returned, ("r", "r", "r"))
        middle = (components(returned, 8), components(returned, 12))
        move_ring(returned, TOP_PATH)
        after = (components(returned, 8), components(returned, 12))
        print(
            "CARGO", cycles, "hub_phase", phase,
            "before", before, "middle", middle, "top", after,
            "avatar", avatar(returned),
            "goals", solid_goals(returned),
            "level", returned.levels_completed,
            flush=True,
        )


arena.run_program("dc22", run)
