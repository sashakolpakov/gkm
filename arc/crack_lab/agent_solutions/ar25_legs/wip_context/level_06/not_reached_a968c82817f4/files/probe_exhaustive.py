import importlib.util
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G


assert not G._workspace_taint_reason(os.getcwd())
spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def replay(env, path):
    node = env.clone()
    for action in path:
        if node.terminal():
            break
        node.step(action)
    return node


def probe(env):
    solver.solve(env)
    base = env.levels_completed
    root = replay(env, [5, 5, 5])
    q = deque([(root, ())])
    seen = set()
    paths = []
    while q and len(seen) < 420:
        node, path = q.popleft()
        key = np.asarray(node.frame())[:63, :63].tobytes()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
        if len(path) >= 22:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            q.append((child, path + (action,)))
    print("P3_STATES", len(paths))
    for n in (5,):
        for down in (1, 4):
            for left in (10, 15):
                for p3 in paths:
                    path = ([5] + [4] * n + [5] + [2] * down + [3] * left
                            + [5] + list(p3) + [5] + [2] * 22)
                    node = replay(env, path)
                    if node.levels_completed > base:
                        print("FOUND", path)
                        return
    print("FOUND", None)


levels, path, err = A.run_program("ar25", probe)
print("END", levels, len(path), err)
