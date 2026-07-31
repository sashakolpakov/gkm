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

RIGHT_EDGE = [
    3, 3, 3, 3, 3,
    (6, 56, 8), (6, 56, 8), (6, 56, 8), (6, 56, 8),
    2, 2, 3, 3, 3, 2, 3, (6, 56, 8), 1, (6, 56, 8), 1, 1,
    (6, 51, 25),
] + [1] * 17 + [3] + [2] * 11 + [1, 1, 1, 4, 4, (6, 51, 25), 4, 4, 4]


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    cells = list(zip(*((env.frame() == 14).nonzero())))
    return None if not cells else (int(cells[0][0] // 2), int(cells[0][1] // 2))


def transitions(root, action, turns=24):
    child = root.clone()
    out = []
    previous = avatar_tile(child)
    for turn in range(1, turns + 1):
        child.step(action)
        current = avatar_tile(child)
        if current != previous or child.levels_completed > 5:
            out.append((turn, current, child.levels_completed))
            previous = current
        if child.levels_completed > 5:
            break
    return out


def run(env):
    solver.solve(env)
    apply(env, RIGHT_EDGE)
    print("ROOT", avatar_tile(env), "level", env.levels_completed)
    phased = env.clone()
    for phase in range(8):
        print(
            "PHASE", phase,
            "u", transitions(phased, 1),
            "d", transitions(phased, 2),
            "l", transitions(phased, 3),
            "r", transitions(phased, 4),
        )
        phased.step(6, 51, 25)

    queue = deque([(env.clone(), [])])
    paths = {avatar_tile(env): []}
    while queue and len(paths) < 160:
        node, path = queue.popleft()
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            position = avatar_tile(child)
            if position not in paths:
                paths[position] = path + [action]
                queue.append((child, path + [action]))
    frontiers = [
        (position, paths[position])
        for position in sorted(paths)
        if position[0] <= 12 or position[1] >= 10
    ]
    print("REACH", len(paths), "bounds", (min(paths), max(paths)), "frontiers", frontiers)


A.run_program("dc22", run)
