import heapq
import importlib.util
import itertools
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

CONTROLS = (
    (6, 46, 22),
    (6, 56, 22),
    (6, 50, 28),
    (6, 56, 28),
    (6, 52, 42),
    (6, 52, 46),
)
ACTIONS = (1, 2, 3, 4) + CONTROLS
TARGET = (50, 10)


def key(env):
    return np.asarray(env.frame())[4:56, :38].tobytes()


def avatar(env):
    ys, xs = np.where(np.asarray(env.frame()) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def distance(env):
    pos = avatar(env)
    return 999 if pos is None else abs(pos[0] - TARGET[0]) + abs(pos[1] - TARGET[1])


def probe(env):
    solver.solve(env)
    base_level = int(env.levels_completed)
    serial = itertools.count()
    root = env.clone()
    root_key = key(root)
    # Prefer dense progress strongly, then shorter paths.
    q = [(distance(root), 0, next(serial), root, root_key)]
    parent = {root_key: (None, None)}
    seen = {root_key}
    best = distance(root)
    expanded = 0
    max_states = 25000
    max_depth = 80

    def path_to(node_key, final_action=None):
        path = [] if final_action is None else [final_action]
        while parent[node_key][0] is not None:
            node_key, action = parent[node_key]
            path.append(action)
        return list(reversed(path))

    while q and len(parent) < max_states:
        _, depth, _, node, node_key = heapq.heappop(q)
        expanded += 1
        dist = distance(node)
        if dist < best:
            best = dist
            print("DENSE", best, avatar(node), "DEPTH", depth,
                  "STATES", len(parent), "PATH", path_to(node_key))
        if depth >= max_depth:
            continue
        for action in ACTIONS:
            child = node.clone()
            child.step(*action) if isinstance(action, tuple) else child.step(action)
            if int(child.levels_completed) > base_level:
                path = path_to(node_key, action)
                print("FOUND", len(path), "STATES", len(parent), "PATH", path)
                return
            child_key = key(child)
            child_depth = depth + 1
            if child_key in seen:
                continue
            seen.add(child_key)
            parent[child_key] = (node_key, action)
            score = distance(child)
            heapq.heappush(
                q, (score, child_depth, next(serial), child, child_key))
        if expanded % 1000 == 0:
            print("PROGRESS", expanded, "STATES", len(parent),
                  "QUEUE", len(q), "BEST", best, "DEPTH", depth)
    print("NOTFOUND", len(parent), "EXPANDED", expanded, "QUEUE", len(q),
          "BEST", best)


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
