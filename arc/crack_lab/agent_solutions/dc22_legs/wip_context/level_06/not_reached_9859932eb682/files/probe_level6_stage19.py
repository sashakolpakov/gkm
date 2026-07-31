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

B = (6, 51, 25)
MACROS = [
    ("u", (1,)), ("d", (2,)), ("l", (3,)), ("r", (4,)),
    ("bu", (B, 1)), ("bd", (B, 2)), ("bl", (B, 3)), ("br", (B, 4)),
]


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    cells = list(zip(*((env.frame() == 14).nonzero())))
    return None if not cells else (int(cells[0][0] // 2), int(cells[0][1] // 2))


def run(env):
    solver.solve(env)
    apply(env, RETURNED)
    queue = deque([(env.clone(), [])])
    seen = {env.frame()[:63].tobytes()}
    positions = {avatar_tile(env)}
    while queue and len(seen) < 3000:
        node, path = queue.popleft()
        if len(path) >= 50:
            continue
        for label, actions in MACROS:
            child = node.clone()
            for action in actions:
                child.step(*action) if isinstance(action, tuple) else child.step(action)
            child_path = path + [label]
            if child.levels_completed > 5:
                print("WIN", child_path, "states", len(seen), flush=True)
                return
            key = child.frame()[:63].tobytes()
            if key in seen:
                continue
            seen.add(key)
            position = avatar_tile(child)
            if position not in positions:
                positions.add(position)
                if position[0] <= 7 or position[1] >= 10 or position[0] >= 21:
                    print("FRONTIER", position, child_path, flush=True)
            queue.append((child, child_path))
        if len(seen) % 250 < 8:
            print("PROGRESS", len(seen), "queue", len(queue), flush=True)
    print(
        "DONE", len(seen), "queue", len(queue), "positions", len(positions),
        "bounds", (min(positions), max(positions)), flush=True,
    )


A.run_program("dc22", run)
