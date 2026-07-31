import importlib.util
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


R = (6, 56, 28)
S = (6, 46, 28)
T = (6, 52, 19)
PREFIX = [
    2, 4, R, 4, R, 4, R, 4, R, 4, 4, 4, 4, 4, 4,
    2, 2, 2, 2, 4,
    R, R, R, R,
    S, S, S, S, S,
    3, 3, 3, 3, 2, 2, T, S,
]
ACTIONS = (1, 2, 3, 4, S, R, T)


def avatar(env):
    ys, xs = np.where(np.asarray(env.frame()) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    solver.solve(env)
    base = int(env.levels_completed)
    for action in PREFIX:
        step(env, action)
    print("ROOT", avatar(env))
    q = deque([(env.clone(), [])])
    seen = {np.asarray(env.frame())[8:56, :38].tobytes()}
    best = 999
    while q and len(seen) < 3500:
        node, path = q.popleft()
        pos = avatar(node)
        if pos is not None:
            distance = abs(pos[0] - 44) + abs(pos[1] - 30)
            if distance < best:
                best = distance
                print("DENSE", best, pos, len(path), path)
        if len(path) >= 50:
            continue
        for action in ACTIONS:
            child = node.clone()
            step(child, action)
            child_path = path + [action]
            if child.levels_completed > base:
                print("FOUND", child_path, "STATES", len(seen))
                return
            key = np.asarray(child.frame())[8:56, :38].tobytes()
            if key not in seen:
                seen.add(key)
                q.append((child, child_path))
    print("FAIL", len(seen), len(q), best)


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
