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

FIRST_TRANSFER = [
    3, 3, 3, 3, 3,
    (6, 56, 8), (6, 56, 8), (6, 56, 8), (6, 56, 8),
    2, 2, 3, 3, 3, 2, 3, (6, 56, 8), 1, (6, 56, 8), 1, 1,
]
SECOND_TRANSFER = [(6, 51, 25)] + [1] * 17 + [3]


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    cells = list(zip(*((env.frame() == 14).nonzero())))
    return None if not cells else (int(cells[0][0] // 2), int(cells[0][1] // 2))


def reachable(root, limit=120):
    queue = deque([(root.clone(), [])])
    seen = {avatar_tile(root)}
    while queue and len(seen) < limit:
        node, path = queue.popleft()
        if node.levels_completed > 5:
            return seen, path
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = avatar_tile(child)
            if key not in seen:
                seen.add(key)
                queue.append((child, path + [action]))
    return seen, None


def assembly_blobs(env):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(1, 8, 10, 12, 14), min_area=2)
        if blob.bbox[1] < 40
    ]


def run(env):
    solver.solve(env)
    apply(env, FIRST_TRANSFER + SECOND_TRANSFER)
    print("STAGE3_ROOT", avatar_tile(env), "level", env.levels_completed)
    print("BLOBS", assembly_blobs(env))

    phased = env.clone()
    for phase in range(8):
        positions, win = reachable(phased)
        print(
            "PHASE", phase,
            "avatar", avatar_tile(phased),
            "reach", sorted(positions),
            "win", win,
            "blobs", assembly_blobs(phased),
        )
        before = phased.frame()
        phased.step(6, 51, 48)
        delta = frame_delta(before, phased.frame())
        print("CLICK", phase, {k: v for k, v in delta.items() if k != "samples"})


A.run_program("dc22", run)
