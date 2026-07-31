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


def movement_reach(root, limit=120):
    queue = deque([root.clone()])
    positions = {avatar_tile(root)}
    won = False
    while queue and len(positions) < limit:
        node = queue.popleft()
        for direction in (1, 2, 3, 4):
            child = node.clone()
            child.step(direction)
            if child.levels_completed > 5:
                won = True
                break
            position = avatar_tile(child)
            if position not in positions:
                positions.add(position)
                queue.append(child)
        if won:
            break
    return positions, won


def run(env):
    solver.solve(env)
    apply(env, HUB)
    islands = {
        0: {(row, col) for row in range(24, 27) for col in range(16, 19)},
        2: {(row, col) for row in range(2, 6) for col in range(2, 6)},
    }
    for labels in RING_PATHS:
        configured = env.clone()
        configure_ring(configured, labels)
        results = []
        for endpoint_phase in (0, 2):
            endpoint = configured.clone()
            endpoint.step(*B_CONTROL)
            for _ in range((endpoint_phase - 3) % 4):
                endpoint.step(*S_CONTROL)
            endpoint.step(*B_CONTROL)
            positions, won = movement_reach(endpoint)
            novel = sorted(
                position for position in positions
                if position is not None and position not in islands[endpoint_phase]
            )
            if won or novel:
                results.append((endpoint_phase, won, novel))
        print("CONFIG", labels, "results", results, flush=True)
        if any(result[1] for result in results):
            return


levels, path, error = A.run_program("dc22", run)
print("HARNESS", levels, len(path), error)
