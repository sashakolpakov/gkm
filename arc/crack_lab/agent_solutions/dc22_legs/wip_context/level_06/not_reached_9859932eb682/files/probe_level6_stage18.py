import importlib.util
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from perception import frame_delta


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
S = (6, 51, 48)


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    cells = list(zip(*((env.frame() == 14).nonzero())))
    return None if not cells else (int(cells[0][0] // 2), int(cells[0][1] // 2))


def run(env):
    solver.solve(env)
    apply(env, RETURNED)
    phased = env.clone()
    for phase in range(4):
        queue = deque([(phased.clone(), [])])
        nodes = {avatar_tile(phased): (phased.clone(), [])}
        while queue and len(nodes) < 100:
            node, path = queue.popleft()
            for direction in (1, 2, 3, 4):
                child = node.clone()
                child.step(direction)
                position = avatar_tile(child)
                if position not in nodes:
                    child_path = path + [direction]
                    nodes[position] = (child, child_path)
                    queue.append((child, child_path))
        print("PHASE", phase, "REACH", len(nodes), sorted(nodes))
        for position in sorted(nodes):
            node, path = nodes[position]
            for direction in (1, 2, 3, 4):
                plain = node.clone()
                plain.step(direction)
                primed = node.clone()
                primed.step(*B)
                primed.step(direction)
                if (
                    avatar_tile(primed) != avatar_tile(plain)
                    or primed.levels_completed != plain.levels_completed
                ):
                    delta = frame_delta(plain.frame()[:63], primed.frame()[:63])
                    print(
                        "TRIGGER", phase, position, "path", path, "dir", direction,
                        "plain", avatar_tile(plain), "primed", avatar_tile(primed),
                        "delta", (delta["count"], delta["bbox"]),
                    )
        phased.step(*S)


A.run_program("dc22", run)
