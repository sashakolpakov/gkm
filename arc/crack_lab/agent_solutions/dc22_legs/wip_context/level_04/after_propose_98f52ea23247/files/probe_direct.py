import importlib.util
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


STAGE = [
    2, 4,
    (6, 56, 28), 4, (6, 56, 28), 4, (6, 56, 28), 4, (6, 56, 28),
    4, 4, 4, 4, 4, 4,
    2, 2, 2, 2, 4,
]
ACTIONS = (1, 2, 3, 4, (6, 52, 19), (6, 46, 28), (6, 56, 28))


def key(env):
    return np.asarray(env.frame())[8:56, :38].tobytes()


def avatar(env):
    ys, xs = np.where(np.asarray(env.frame()) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def probe(env):
    solver.solve(env)
    base_level = int(env.levels_completed)
    for action in STAGE:
        env.step(*action) if isinstance(action, tuple) else env.step(action)
    q = deque([(env.clone(), [])])
    seen = {key(env)}
    best = 999
    while q and len(seen) < 2500:
        node, path = q.popleft()
        pos = avatar(node)
        if pos:
            distance = abs(pos[0] - 44) + abs(pos[1] - 30)
            if distance < best:
                best = distance
                print("DENSE", best, pos, len(path), path)
        if len(path) >= 70:
            continue
        for action in ACTIONS:
            try:
                child = node.clone()
                child.step(*action) if isinstance(action, tuple) else child.step(action)
            except IndexError:
                continue
            child_path = path + [action]
            if child.levels_completed > base_level:
                print("FOUND", STAGE + child_path, "TAIL", child_path, "STATES", len(seen))
                return
            child_key = key(child)
            if child_key not in seen:
                seen.add(child_key)
                q.append((child, child_path))
    print("NOTFOUND", len(seen), len(q), "BEST", best)


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
