import importlib.util
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G


taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"TAINTED WORKSPACE: {taint_reason}")

spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

REMOTE = [
    3, 3, 3, 3, 3,
    (6, 56, 8), (6, 56, 8), (6, 56, 8), (6, 56, 8),
    2, 2, 3, 3, 3, 2, 3, (6, 56, 8), 1, (6, 56, 8), 1, 1,
    (6, 51, 25),
] + [1] * 17 + [3] + [2] * 11

B = (6, 51, 25)


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    cells = list(zip(*((env.frame() == 14).nonzero())))
    return None if not cells else (int(cells[0][0] // 2), int(cells[0][1] // 2))


def synchronized(root, direction, turns=20):
    child = root.clone()
    out = []
    previous = avatar_tile(child)
    for turn in range(1, turns + 1):
        child.step(*B)
        child.step(direction)
        current = avatar_tile(child)
        if current != previous or child.levels_completed > 5:
            out.append((turn, current, child.levels_completed))
            previous = current
        if child.levels_completed > 5:
            break
    return out


def run(env):
    solver.solve(env)
    apply(env, REMOTE)
    for name, edge_path in (
        ("LEFT", [1, 1, 1, 3, B, 3, 3, 3]),
        ("RIGHT", [1, 1, 1, 4, 4, B, 4, 4, 4]),
    ):
        edge = env.clone()
        apply(edge, edge_path)
        print("ROOT", name, avatar_tile(edge))
        for direction in (1, 2, 3, 4):
            print(name, direction, synchronized(edge, direction))


A.run_program("dc22", run)
