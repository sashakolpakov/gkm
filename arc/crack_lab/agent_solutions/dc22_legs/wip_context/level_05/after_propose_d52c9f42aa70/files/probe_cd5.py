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


def crop_key(env):
    return np.asarray(env.frame())[4:56, :38].tobytes()


def summary(env):
    a = np.asarray(env.frame())
    targets = int((a[4:56, :38] == 15).sum())
    ys, xs = np.where(a[4:56, :38] == 8)
    bbox = None if not len(ys) else (
        int(ys.min() + 4), int(xs.min()), int(ys.max() + 4), int(xs.max()))
    return targets, bbox, int(env.levels_completed)


def probe(env):
    solver.solve(env)
    root = env.clone()
    root_key = crop_key(root)
    queue = deque([(root, root_key, "")])
    seen = {root_key}
    best = summary(root)[0]
    print("ROOT", summary(root))
    while queue and len(seen) < 200:
        node, node_key, path = queue.popleft()
        for label, action in (("C", C), ("D", D)):
            child = node.clone()
            child.step(*action)
            child_key = crop_key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            child_path = path + label
            state = summary(child)
            best = min(best, state[0])
            print("STATE", len(seen), child_path, state)
            queue.append((child, child_key, child_path))
    print("DONE", len(seen), "BEST", best)


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
