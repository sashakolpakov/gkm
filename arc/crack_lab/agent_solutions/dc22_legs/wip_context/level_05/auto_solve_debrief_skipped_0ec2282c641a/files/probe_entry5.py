import importlib.util
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

C = (6, 50, 28)
D = (6, 56, 28)
DOCK = (6, 52, 35)
X = (6, 44, 29)
Y = (6, 54, 29)
Z = (6, 60, 29)
E = (6, 52, 42)
PREFIX = (C, C, C, D, D, D, DOCK, X, X, X,
          Z, Z, Z, Y, Y, Y, 3, E, 3, 1, 1, 3)


def key(env):
    return np.asarray(env.frame())[:56, :38].tobytes()


def avatar(env):
    ys, xs = np.where(np.asarray(env.frame()) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def probe(env):
    solver.solve(env)
    positions = []
    for action in PREFIX:
        env.step(*action) if isinstance(action, tuple) else env.step(action)
        positions.append(avatar(env))
    print("ENTRY", positions, int(env.levels_completed))
    root = env.clone()
    queue = deque([(root, [])])
    seen = {key(root)}
    reached = {}
    while queue and len(seen) < 200:
        node, path = queue.popleft()
        pos = avatar(node)
        if pos not in reached:
            reached[pos] = path
            print("REACH", pos, path)
        if len(path) >= 20:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_key = key(child)
            if child_key not in seen:
                seen.add(child_key)
                queue.append((child, path + [action]))
    print("DONE", len(seen), len(reached))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
