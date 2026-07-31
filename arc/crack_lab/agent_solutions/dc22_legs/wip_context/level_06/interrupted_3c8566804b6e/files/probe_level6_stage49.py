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
    "b": (1, B_CONTROL, 2),
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


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def ring(env):
    blobs = [
        blob for blob in connected_components(env.frame(), colors=(8,), min_area=8)
        if blob.bbox[1] < 40
    ]
    if not blobs:
        return None
    blob = max(blobs, key=lambda item: item.area)
    return blob.bbox, blob.area


def avatar_tile(env):
    blobs = connected_components(env.frame(), colors=(14,), min_area=2)
    if not blobs:
        return None
    blob = max(blobs, key=lambda item: item.area)
    return blob.bbox[0] // 2, blob.bbox[1] // 2


def phase1_result(root):
    child = root.clone()
    apply(child, [B_CONTROL, S_CONTROL, S_CONTROL, B_CONTROL])
    return avatar_tile(child), child.levels_completed


def phase1_combinations(root, path):
    for a_phase in range(6):
        for b_phase in range(2):
            child = root.clone()
            for _ in range(a_phase):
                child.step(*A_CONTROL)
            for _ in range(b_phase):
                apply(child, [1, B_CONTROL, 2])
            result = phase1_result(child)
            if result != ((24, 9), 5):
                print(
                    "ALIGNED_PHASE1", path,
                    (a_phase, b_phase), result,
                )


def run(env):
    solver.solve(env)
    apply(env, HUB)
    queue = deque([(env.clone(), [])])
    seen = {env.frame()[:63].tobytes()}
    configs = {ring(env): []}
    print("PHASE1", [], phase1_result(env))
    phase1_combinations(env, [])
    while queue and len(seen) < 500:
        node, path = queue.popleft()
        if len(path) >= 20:
            continue
        for label, (outward, control, inward) in D_CONTROLS.items():
            child = node.clone()
            apply(child, [outward, control, inward])
            child_path = path + [label]
            if child.levels_completed > 5:
                print("WIN", child_path, "states", len(seen))
                return
            key = child.frame()[:63].tobytes()
            if key in seen:
                continue
            seen.add(key)
            config = ring(child)
            if config not in configs:
                configs[config] = child_path
                print("RING", config, child_path)
                result = phase1_result(child)
                if result != ((24, 9), 5):
                    print("PHASE1", child_path, result)
                phase1_combinations(child, child_path)
            queue.append((child, child_path))
    print("DONE", len(seen), "queue", len(queue), "configs", configs)


A.run_program("dc22", run)
