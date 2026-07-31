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
REVERSE = [
    B_CONTROL, 4,
    2, 2, 2, 3, 2, 3, 3, 2, 3, 3, 1, A_CONTROL, 1,
] + [1] * 7 + [3]
RING_PATHS = [
    [],
    ["u"], ["d"], ["l"], ["r"],
    ["u", "l"], ["u", "r"], ["d", "l"], ["d", "r"],
    ["l", "l"], ["u", "r", "u"], ["l", "l", "l"],
    ["u", "r", "u", "u"], ["l", "l", "l", "l"],
    ["u", "r", "u", "u", "l"],
    ["u", "r", "u", "u", "l", "l"],
    ["u", "r", "u", "u", "l", "l", "u"],
    ["u", "r", "u", "u", "l", "l", "u", "u"],
    ["u", "r", "u", "u", "l", "l", "u", "u", "u"],
]


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    blobs = connected_components(env.frame(), colors=(14,), min_area=2)
    if not blobs:
        return None
    blob = max(blobs, key=lambda item: item.area)
    return blob.bbox[0] // 2, blob.bbox[1] // 2


def configure_ring(env, labels):
    for label in labels:
        outward, control, inward = D_CONTROLS[label]
        apply(env, [outward, control, inward])


def search(root):
    actions = [("u", 1), ("d", 2), ("l", 3), ("r", 4), ("b", B_CONTROL)]
    queue = deque([(root.clone(), [])])
    seen = {root.frame()[:63].tobytes()}
    positions = {avatar_tile(root)}
    while queue and len(seen) < 240:
        node, path = queue.popleft()
        if len(path) >= 35:
            continue
        for label, action in actions:
            child = node.clone()
            child.step(*action) if isinstance(action, tuple) else child.step(action)
            child_path = path + [label]
            if child.levels_completed > 5:
                return "win", child_path, len(seen)
            key = child.frame()[:63].tobytes()
            if key in seen:
                continue
            seen.add(key)
            positions.add(avatar_tile(child))
            queue.append((child, child_path))
    visible = sorted(position for position in positions if position is not None)
    return "done", visible, len(seen)


def run(env):
    solver.solve(env)
    apply(env, HUB)
    for labels in RING_PATHS:
        child = env.clone()
        configure_ring(child, labels)
        apply(child, REVERSE)
        result = search(child)
        novel = []
        if result[0] == "done":
            novel = [
                position for position in result[1]
                if position[1] >= 10 or position[0] <= 7
            ]
        print("CONFIG", labels, result[0], result[2], "novel", novel, flush=True)
        if result[0] == "win":
            print("WIN", labels, result[1], flush=True)
            return


levels, path, error = A.run_program("dc22", run)
print("HARNESS", levels, len(path), error)
