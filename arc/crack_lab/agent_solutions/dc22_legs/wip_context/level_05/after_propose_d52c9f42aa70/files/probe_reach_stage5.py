import importlib.util
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

A_CTL = (6, 46, 22)
B_CTL = (6, 56, 22)
C = (6, 50, 28)
D = (6, 56, 28)
E = (6, 52, 42)
F = (6, 52, 46)


def key(env):
    return np.asarray(env.frame())[4:56, :38].tobytes()


def avatar(env):
    ys, xs = np.where(np.asarray(env.frame()) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def probe(env):
    solver.solve(env)
    for action in (C, C, C, D):
        env.step(*action)
    actions = (1, 2, 3, 4, A_CTL, B_CTL, E, F)
    root = env.clone()
    root_key = key(root)
    queue = deque([(root, root_key, [])])
    seen = {root_key}
    positions = {}
    while queue and len(seen) < 1200:
        node, node_key, path = queue.popleft()
        pos = avatar(node)
        if pos not in positions:
            positions[pos] = path
            print("REACH", pos, len(path), path)
        if len(path) >= 18:
            continue
        for action in actions:
            child = node.clone()
            child.step(*action) if isinstance(action, tuple) else child.step(action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, child_key, path + [action]))
    print("DONE", len(seen), len(queue), len(positions))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
