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

RETURNED = [
    3, 3, 3, 3, 3,
    (6, 56, 8), (6, 56, 8), (6, 56, 8), (6, 56, 8),
    2, 2, 3, 3, 3, 2, 3, (6, 56, 8), 1, (6, 56, 8), 1, 1,
    (6, 51, 25),
] + [1] * 17 + [3] + [2] * 11 + [1]

B = (6, 51, 25)


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    cells = list(zip(*((env.frame() == 14).nonzero())))
    return None if not cells else (int(cells[0][0] // 2), int(cells[0][1] // 2))


def pair_tiles(env, wanted=(11, 12)):
    frame = env.frame()
    out = []
    for row in range(31):
        for col in range(20):
            pair = tuple(sorted(set(int(v) for v in frame[
                2 * row:2 * row + 2, 2 * col:2 * col + 2
            ].ravel())))
            if pair == wanted:
                out.append((row, col))
    return out


def run(env):
    solver.solve(env)
    apply(env, RETURNED)
    print("ROOT", avatar_tile(env), "pair", pair_tiles(env))
    for direction in (1, 2, 3, 4):
        child = env.clone()
        child.step(*B)
        child.step(direction)
        print(
            "ONE", direction, "avatar", avatar_tile(child),
            "pair", pair_tiles(child), "level", child.levels_completed,
        )
    for direction in (1, 2, 3, 4):
        child = env.clone()
        trace = []
        previous = pair_tiles(child)
        for turn in range(1, 17):
            child.step(*B)
            child.step(direction)
            current = pair_tiles(child)
            if current != previous or child.levels_completed > 5:
                trace.append((turn, current, avatar_tile(child), child.levels_completed))
                previous = current
            if child.levels_completed > 5:
                break
        print("REPEAT", direction, trace)


A.run_program("dc22", run)
