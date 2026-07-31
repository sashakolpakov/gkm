import importlib.util
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G


taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"TAINTED WORKSPACE: {taint_reason}")

spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

RETURNED = [
    3, 3, 3, 3, 3,
    (6, 56, 8), (6, 56, 8), (6, 56, 8), (6, 56, 8),
    2, 2, 3, 3, 3, 2, 3, (6, 56, 8), 1, (6, 56, 8), 1, 1,
    (6, 51, 25),
] + [1] * 17 + [3] + [2] * 11 + [1]


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    cells = list(zip(*((env.frame() == 14).nonzero())))
    return None if not cells else (int(cells[0][0] // 2), int(cells[0][1] // 2))


def reachable(root, limit=100):
    queue = deque([(root.clone(), [])])
    paths = {avatar_tile(root): []}
    while queue and len(paths) < limit:
        node, path = queue.popleft()
        if node.levels_completed > 5:
            return paths, path
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = avatar_tile(child)
            if key not in paths:
                paths[key] = path + [action]
                queue.append((child, path + [action]))
    return paths, None


def run(env):
    solver.solve(env)
    apply(env, RETURNED)
    phased = env.clone()
    for phase in range(5):
        paths, win = reachable(phased)
        positions = sorted(paths)
        novel = [
            (position, paths[position])
            for position in positions
            if position[0] <= 6 or position[0] >= 22 or 7 <= position[1] < 20
        ]
        print("PHASE", phase, "avatar", avatar_tile(phased),
              "count", len(positions), "bounds", (min(positions), max(positions)),
              "novel", novel, "win", win)
        phased.step(6, 51, 25)


A.run_program("dc22", run)
