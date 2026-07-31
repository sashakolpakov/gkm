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

A_CONTROL = (6, 56, 8)
B_CONTROL = (6, 51, 25)
S_CONTROL = (6, 51, 48)
REMOTE = [
    3, 3, 3, 3, 3,
    A_CONTROL, A_CONTROL, A_CONTROL, A_CONTROL,
    2, 2, 3, 3, 3, 2, 3, A_CONTROL, 1, A_CONTROL, 1, 1,
    B_CONTROL,
] + [1] * 17 + [3] + [2] * 11
STAGED = (
    REMOTE
    + [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
)


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    frame = env.frame()
    for row in range(31):
        for col in range(32):
            if (frame[2 * row:2 * row + 2, 2 * col:2 * col + 2] == 14).all():
                return row, col
    return None


def run(env):
    solver.solve(env)
    apply(env, STAGED)
    phased = env.clone()
    for phase in range(4):
        queue = deque([(phased.clone(), [])])
        nodes = {avatar_tile(phased): (phased.clone(), [])}
        while queue:
            node, path = queue.popleft()
            for direction in (1, 2, 3, 4):
                child = node.clone()
                child.step(direction)
                position = avatar_tile(child)
                if position not in nodes:
                    child_path = path + [direction]
                    nodes[position] = (child, child_path)
                    queue.append((child, child_path))
        print("PHASE", phase, "REACH", sorted(nodes))
        for position in sorted(nodes):
            node, path = nodes[position]
            for name, control in (("A", A_CONTROL), ("B", B_CONTROL)):
                for direction in (1, 2, 3, 4):
                    plain = node.clone()
                    plain.step(direction)
                    primed = node.clone()
                    primed.step(*control)
                    primed.step(direction)
                    if (
                        avatar_tile(primed) != avatar_tile(plain)
                        or primed.levels_completed != plain.levels_completed
                    ):
                        print(
                            "TRIGGER", phase, position, path, name, direction,
                            "plain", avatar_tile(plain),
                            "primed", avatar_tile(primed),
                            "level", primed.levels_completed,
                        )
        phased.step(*S_CONTROL)


A.run_program("dc22", run)
