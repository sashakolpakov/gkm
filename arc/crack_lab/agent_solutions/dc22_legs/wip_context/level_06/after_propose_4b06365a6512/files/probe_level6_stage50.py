import importlib.util
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from perception import connected_components


taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"TAINTED WORKSPACE: {taint_reason}")

spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

A_CONTROL = (6, 56, 8)
B_CONTROL = (6, 51, 25)
S_CONTROL = (6, 51, 48)
D_CONTROLS = {
    "u": (1, (6, 50, 32), 2),
    "d": (2, (6, 50, 40), 1),
    "l": (3, (6, 46, 36), 4),
    "r": (4, (6, 54, 36), 3),
}
REMOTE = [
    3, 3, 3, 3, 3,
    A_CONTROL, A_CONTROL, A_CONTROL, A_CONTROL,
    2, 2, 3, 3, 3, 2, 3, A_CONTROL, 1, A_CONTROL, 1, 1,
    B_CONTROL,
] + [1] * 17 + [3] + [2] * 11
HUB = (
    REMOTE
    + [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
    + [S_CONTROL, S_CONTROL, S_CONTROL, 3, B_CONTROL]
)
RING_PATH = ["u", "r", "u", "u", "l", "l", "u", "u", "u"]


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    blobs = connected_components(env.frame(), colors=(14,), min_area=2)
    if not blobs:
        return None
    blob = max(blobs, key=lambda item: item.area)
    return blob.bbox[0] // 2, blob.bbox[1] // 2


def run(env):
    solver.solve(env)
    apply(env, HUB)
    for label in RING_PATH:
        outward, control, inward = D_CONTROLS[label]
        apply(env, [outward, control, inward])
    apply(env, [B_CONTROL, S_CONTROL, S_CONTROL, S_CONTROL, B_CONTROL])
    print("ROOT", avatar_tile(env), "level", env.levels_completed, flush=True)
    actions = [("u", 1), ("d", 2), ("l", 3), ("r", 4), ("b", B_CONTROL)]
    queue = deque([(env.clone(), [])])
    seen = {env.frame()[:63].tobytes()}
    positions = {avatar_tile(env)}
    while queue and len(seen) < 800:
        node, path = queue.popleft()
        if len(path) >= 40:
            continue
        for label, action in actions:
            child = node.clone()
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
                print("NEW", position, child_path, flush=True)
            queue.append((child, child_path))
    print("DONE", len(seen), "queue", len(queue), "positions", sorted(positions), flush=True)


A.run_program("dc22", run)
