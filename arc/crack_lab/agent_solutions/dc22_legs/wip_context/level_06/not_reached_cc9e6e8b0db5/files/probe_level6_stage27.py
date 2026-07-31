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
    for phase in (0, 2, 3):
        root = env.clone()
        for _ in range(phase):
            root.step(*S_CONTROL)
        root.step(3)
        root.step(*B_CONTROL)
        queue = deque([root.clone()])
        nodes = {avatar_tile(root): root.clone()}
        while queue:
            node = queue.popleft()
            for direction in (1, 2, 3, 4):
                child = node.clone()
                child.step(direction)
                position = avatar_tile(child)
                if position not in nodes:
                    nodes[position] = child
                    queue.append(child)
        print("PHASE", phase, "ROOT", avatar_tile(root), "REACH", sorted(nodes))
        for position in sorted(nodes):
            node = nodes[position]
            top_before = node.frame()[4:12, 32:40].copy()
            child = node.clone()
            for a_phase in range(1, 7):
                child.step(*A_CONTROL)
                top_changed = bool((child.frame()[4:12, 32:40] != top_before).any())
                if top_changed or child.levels_completed > 5:
                    print(
                        "REVEAL", phase, position, "A", a_phase,
                        "avatar", avatar_tile(child),
                        "level", child.levels_completed,
                    )
            toggled = node.clone()
            before = toggled.frame().copy()
            toggled.step(*B_CONTROL)
            delta = frame_delta(before[:63], toggled.frame()[:63])
            if delta["count"]:
                print(
                    "B_DELTA", phase, position,
                    (delta["count"], delta["bbox"]),
                    "top", bool((toggled.frame()[4:12, 32:40] != top_before).any()),
                    "level", toggled.levels_completed,
                )
            for direction in (1, 2, 3, 4):
                plain = node.clone()
                plain.step(direction)
                primed = node.clone()
                primed.step(*B_CONTROL)
                primed.step(direction)
                if (
                    avatar_tile(primed) != avatar_tile(plain)
                    or primed.levels_completed != plain.levels_completed
                ):
                    print(
                        "B_TRIGGER", phase, position, direction,
                        "plain", avatar_tile(plain),
                        "primed", avatar_tile(primed),
                        "level", primed.levels_completed,
                    )


A.run_program("dc22", run)
