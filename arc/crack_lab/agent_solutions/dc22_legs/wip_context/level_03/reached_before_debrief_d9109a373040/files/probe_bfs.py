import importlib.util
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


MOVES = (1, 2, 3, 4)
CONTROLS = ((6, 51, 18), (6, 51, 27), (6, 51, 36))
ACTIONS = MOVES + CONTROLS


def key(env, path=()):
    return np.asarray(env.frame())[8:56, :38].tobytes()


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def probe(env):
    solver.solve(env)
    base_level = int(env.levels_completed)
    root = env.clone()

    def replay(path):
        node = root.clone()
        try:
            for action in path:
                if node.terminal():
                    return None
                if isinstance(action, tuple):
                    node.step(*action)
                else:
                    node.step(action)
            return node
        except IndexError:
            return None

    q = deque([[]])
    seen = {key(env)}
    best = 999
    positions = set()
    landmarks = {
        "color11": (38, 32),
        "color15": (42, 16),
        "endpoint_a": (18, 6),
        "endpoint_b": (46, 12),
    }
    nearest = {name: (999, None, None) for name in landmarks}
    missing = None
    while q and len(seen) < 6000:
        path = q.popleft()
        node = replay(path)
        if node is None:
            continue
        pos = avatar(node.frame())
        if pos:
            positions.add(pos)
            for name, target in landmarks.items():
                distance = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
                if distance < nearest[name][0]:
                    nearest[name] = (distance, pos, path)
            distance = abs(pos[0] - 38) + abs(pos[1] - 32)
            if distance < best:
                best = distance
                print("DENSE", best, "POS", pos, "DEPTH", len(path), "PATH", path)
        elif missing is None:
            missing = path
        if len(path) >= 90:
            continue
        for action in ACTIONS:
            child_path = path + [action]
            child = replay(child_path)
            if child is None:
                continue
            if child.levels_completed > base_level:
                print("FOUND", child_path, "STATES", len(seen))
                return
            child_key = key(child, child_path)
            if child_key not in seen:
                seen.add(child_key)
                q.append(child_path)
    print("NOTFOUND", len(seen), "QUEUE", len(q), "BEST", best)
    print("POSITIONS", sorted(positions))
    print("NEAREST", nearest)
    print("NO_AVATAR_PATH", missing)


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
