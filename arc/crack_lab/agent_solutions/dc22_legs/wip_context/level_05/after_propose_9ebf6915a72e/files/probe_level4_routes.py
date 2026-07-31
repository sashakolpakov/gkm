import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def avatar(env):
    ys, xs = np.where(np.asarray(env.frame()) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def apply(env, action):
    before = np.asarray(env.frame()).copy()
    env.step(*action) if isinstance(action, tuple) else env.step(action)
    delta = frame_delta(before, env.frame())
    return avatar(env), (delta["count"], delta["bbox"])


def trace(root, name, actions):
    env = root.clone()
    print(name, "S", avatar(env))
    for i, action in enumerate(actions, 1):
        print(name, i, action, apply(env, action), "L", int(env.levels_completed))


def rows(env):
    a = np.asarray(env.frame())
    return tuple(tuple(int(v) for v in a[y, 2:34:2]) for y in (28, 38, 40))


def probe(env):
    solver.solve(env)
    trace(env, "DOWN", [2] * 8)
    trace(env, "TOP_DOWN", [(6, 52, 19)] + [2] * 8)
    trace(env, "SHIFT_DOWN", [(6, 46, 28)] * 4 + [2] * 8)
    trace(env, "LOWER_ROUTE",
          [(6, 46, 28)] * 4 + [2] * 10 + [4] * 12 + [2] * 10 + [4] * 12)
    staged = env.clone()
    for action in [
            2, 4,
            (6, 56, 28), 4, (6, 56, 28), 4, (6, 56, 28), 4, (6, 56, 28),
            4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 4]:
        apply(staged, action)
    for n in range(7):
        probe_state = staged.clone()
        for _ in range(n):
            apply(probe_state, (6, 46, 28))
        print("FERRY", n, avatar(probe_state), rows(probe_state))
        trace(probe_state, f"BOARD{n}", [3] * 8 + [(6, 46, 28)] * 3)
    center = staged.clone()
    for _ in range(5):
        apply(center, (6, 46, 28))
    for _ in range(4):
        apply(center, 3)
    trace(center, "CENTER_DOWN", [2] * 8)
    trace(center, "CENTER_TOGGLE_DOWN", [(6, 52, 19)] + [2] * 8)
    on_switch = center.clone()
    apply(on_switch, 2)
    apply(on_switch, 2)
    trace(on_switch, "ON_SWITCH", [(6, 52, 19)] * 4 + [1, 2, 3, 4])


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
