import importlib.util
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from perception import connected_components, frame_delta


taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"TAINTED WORKSPACE: {taint_reason}")

spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

STAGE3 = [
    3, 3, 3, 3, 3,
    (6, 56, 8), (6, 56, 8), (6, 56, 8), (6, 56, 8),
    2, 2, 3, 3, 3, 2, 3, (6, 56, 8), 1, (6, 56, 8), 1, 1,
    (6, 51, 25),
] + [1] * 17 + [3]


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    cells = list(zip(*((env.frame() == 14).nonzero())))
    return None if not cells else (int(cells[0][0] // 2), int(cells[0][1] // 2))


def path_to(root, target, limit=100):
    queue = deque([(root.clone(), [])])
    seen = {avatar_tile(root)}
    while queue and len(seen) < limit:
        node, path = queue.popleft()
        if avatar_tile(node) == target:
            return path
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = avatar_tile(child)
            if key not in seen:
                seen.add(key)
                queue.append((child, path + [action]))
    return None


def run(env):
    solver.solve(env)
    apply(env, STAGE3)
    teleport_path = path_to(env, (19, 28))
    print("TELEPORT_PATH", teleport_path)
    apply(env, teleport_path)
    print("REMOTE", avatar_tile(env), "level", env.levels_completed)

    base = env.frame()
    print("ACTIVE_CONTROLS")
    tested = set()
    for blob in connected_components(base, min_area=2):
        r0, c0, r1, c1 = blob.bbox
        if c0 < 40 or blob.area > 100:
            continue
        point = ((c0 + c1) // 2, (r0 + r1) // 2)
        if point in tested:
            continue
        tested.add(point)
        child = env.clone()
        child.step(6, *point)
        delta = frame_delta(base, child.frame())
        if delta["count"] > 1:
            print(point, blob.color, blob.bbox, blob.area,
                  {k: v for k, v in delta.items() if k != "samples"})

    positions = {}
    queue = deque([(env.clone(), [])])
    positions[avatar_tile(env)] = []
    while queue and len(positions) < 120:
        node, path = queue.popleft()
        if node.levels_completed > 5:
            print("MOVE_WIN", path)
            return
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = avatar_tile(child)
            if key not in positions:
                positions[key] = path + [action]
                queue.append((child, path + [action]))
    print("REMOTE_REACH", sorted(positions), "level", env.levels_completed)


A.run_program("dc22", run)
