import importlib.util
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

C1 = (6, 51, 18)
C2 = (6, 51, 27)
C3 = (6, 51, 36)
C4 = (6, 51, 45)
ACTIONS = (1, 2, 3, 4, C1, C2, C3, C4)
TO_15 = [3, C1, C2, 3, 3, 3, 2, 2, 2, 3, C1, 3, 3, 3, 3,
         C2, 3, 3, 3, C2, 1, 1, 1, 1, C3, 1, 1, 4, 4]


def do(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def key(env):
    return np.asarray(env.frame())[:63].tobytes()


def probe(env):
    solver.solve(env)
    base_level = int(env.levels_completed)
    for action in TO_15:
        do(env, action)
    root = env.clone()

    def replay(path):
        node = root.clone()
        for action in path:
            do(node, action)
        return node

    q = deque([[]])
    seen = {key(root)}
    best = 999
    while q and len(seen) < 6000:
        path = q.popleft()
        node = replay(path)
        pos = avatar(node.frame())
        if pos is not None:
            distance = abs(pos[0] - 38) + abs(pos[1] - 32)
            if distance < best:
                best = distance
                print("DENSE", best, "POS", pos, "DEPTH", len(path),
                      "PATH", path)
        if len(path) >= 70:
            continue
        for action in ACTIONS:
            child_path = path + [action]
            child = replay(child_path)
            if child.levels_completed > base_level:
                print("FOUND", child_path, "STATES", len(seen))
                return
            child_key = key(child)
            if child_key not in seen:
                seen.add(child_key)
                q.append(child_path)
    print("NOTFOUND", len(seen), "QUEUE", len(q), "BEST", best)


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
