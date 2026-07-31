import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena


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


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, actions):
    for action in actions:
        step(env, action)


def goal_metric(env):
    frame = np.asarray(env.frame())
    tiles = []
    for row in range(31):
        for col in range(20):
            block = frame[2 * row:2 * row + 2, 2 * col:2 * col + 2]
            count = int(np.count_nonzero(block == 11))
            if count:
                tiles.append((count, row, col, tuple(int(v) for v in block.ravel())))
    return sorted(tiles, reverse=True)


def report(label, env):
    print(label, goal_metric(env)[:8], env.levels_completed, flush=True)


def run(env):
    solver.solve(env)
    report("ENTRY", env)
    apply(env, REMOTE)
    report("REMOTE", env)
    apply(env, ROOT_TO_SELECTOR)
    report("SELECTOR", env)

    for control, name, phases in ((A, "A", 6), (B, "B", 4), (S, "S", 4)):
        phased = env.clone()
        for phase in range(phases):
            report(f"{name}{phase}", phased)
            step(phased, control)

    hub = env.clone()
    apply(hub, [S, S, S, 3, B])
    report("HUB", hub)
    routed = hub.clone()
    for label in ("u", "r", "u", "u", "l", "l", "u", "u", "u"):
        apply(routed, D[label])
        report(f"RING_{label}", routed)

    for selector_phase in range(4):
        endpoint = env.clone()
        apply(endpoint, [S] * selector_phase + [3, B])
        report(f"ENDPOINT_{selector_phase}", endpoint)


arena.run_program("dc22", run)
