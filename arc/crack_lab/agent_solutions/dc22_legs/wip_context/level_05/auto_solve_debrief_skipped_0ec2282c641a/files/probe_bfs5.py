import importlib.util
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

E = (6, 52, 42)
F = (6, 52, 46)
C = (6, 50, 28)
D = (6, 56, 28)
CTRL_A = (6, 46, 22)
CTRL_B = (6, 56, 22)
ACTIONS = (1, 2, 3, 4, CTRL_A, CTRL_B, C, D, E, F)
PREFIX = [3, E, 3, 1, 1]


def key(env):
    return np.asarray(env.frame())[4:56, :38].tobytes()


def avatar(env):
    ys, xs = np.where(np.asarray(env.frame()) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def probe(env):
    solver.solve(env)
    base_level = int(env.levels_completed)
    for action in PREFIX:
        if isinstance(action, tuple):
            env.step(*action)
        else:
            env.step(action)
    root = env.clone()
    root_key = key(root)
    q = deque([(root, root_key, 0)])
    parent = {root_key: (None, None)}
    best_regions = set()
    last_report = 0
    found = None
    def path_to(node_key):
        path = []
        while parent[node_key][0] is not None:
            node_key, action = parent[node_key]
            path.append(action)
        return list(reversed(path))

    while q and len(parent) < 7000:
        node, node_key, depth = q.popleft()
        pos = avatar(node)
        if pos not in best_regions:
            best_regions.add(pos)
            print("REACH", pos, "DEPTH", depth, "STATES", len(parent), "PATH", path_to(node_key))
        if depth >= 100:
            continue
        for action in ACTIONS:
            child = node.clone()
            if isinstance(action, tuple):
                child.step(*action)
            else:
                child.step(action)
            if int(child.levels_completed) > base_level:
                found = (node_key, action)
                print("FOUND", depth + 1, "STATES", len(parent))
                q.clear()
                break
            child_key = key(child)
            if child_key in parent:
                continue
            parent[child_key] = (node_key, action)
            q.append((child, child_key, depth + 1))
        if len(parent) - last_report >= 1000:
            last_report = len(parent)
            print("PROGRESS", len(parent), "QUEUE", len(q), "DEPTH", depth)
    if found:
        node_key, final_action = found
        path = [final_action]
        while parent[node_key][0] is not None:
            node_key, action = parent[node_key]
            path.append(action)
        path.reverse()
        print("PATH", path)
    else:
        print("NOTFOUND", len(parent), "QUEUE", len(q), "REGIONS", sorted(best_regions))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
