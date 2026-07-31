import importlib.util
import os
import sys
from collections import deque

import numpy as np

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
D_BY_POSITION = {
    (28, 17): ("DU", (6, 50, 32)),
    (30, 17): ("DD", (6, 50, 40)),
    (29, 16): ("DL", (6, 46, 36)),
    (18, 27): ("DR", (6, 54, 36)),
}


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_position(env):
    frame = env.frame()
    rows, cols = np.where(frame[:62, :40] == 14)
    if len(rows):
        return int(rows.min() // 2), int(cols.min() // 2)
    blobs = connected_components(frame, colors=(14,), min_area=2)
    if not blobs:
        return None
    blob = max(blobs, key=lambda item: item.area)
    return blob.bbox[0] // 2, blob.bbox[1] // 2


def world_key(env):
    return env.frame()[:62, :40].tobytes()


def apply(env, path):
    for action in path:
        step(env, action)


def movement_closure(root):
    opposite = {1: 2, 2: 1, 3: 4, 4: 3}
    by_position = {}
    winning_walk = None

    def visit(path):
        nonlocal winning_walk
        position = avatar_position(root)
        by_position[position] = list(path)
        for direction in (1, 2, 3, 4):
            before = avatar_position(root)
            root.step(direction)
            after = avatar_position(root)
            if root.levels_completed > 5:
                winning_walk = list(path) + [direction]
                return
            moved = after != before
            if moved and after not in by_position:
                visit(path + [direction])
                if winning_walk is not None:
                    return
            if moved:
                root.step(opposite[direction])
                restored = avatar_position(root)
                if restored != before:
                    raise RuntimeError(
                        f"non-reversible walk {before}->{after}->{restored}"
                    )

    visit([])
    return by_position, winning_walk


def canonical(by_position):
    visible = [
        position for position in by_position
        if position is not None and position[1] < 20
    ]
    position = min(visible) if visible else min(
        position for position in by_position if position is not None
    )
    return position, by_position[position]


def run(env):
    solver.solve(env)
    base = env.clone()
    queue = deque([[]])
    seen = set()
    best = (99, -1, 0)
    while queue and len(seen) < 2400:
        root_path = queue.popleft()
        root = base.clone()
        apply(root, root_path)
        if root.levels_completed > 5:
            print("WIN", root_path, "states", len(seen), flush=True)
            return
        by_position, winning_walk = movement_closure(root)
        if winning_walk is not None:
            path = root_path + winning_walk
            print("WIN", path, "states", len(seen), flush=True)
            return
        canonical_position, canonical_walk = canonical(by_position)
        canonical_env = base.clone()
        apply(canonical_env, root_path + canonical_walk)
        key = world_key(canonical_env)
        if key in seen:
            continue
        seen.add(key)

        positions = [
            position for position in by_position if position is not None
        ]
        metric = (
            min(position[0] for position in positions),
            max(position[1] for position in positions),
            len(positions),
        )
        dense = (best[0] - metric[0], metric[1] - best[1], metric[2] - best[2])
        if any(value > 0 for value in dense):
            best = (
                min(best[0], metric[0]),
                max(best[1], metric[1]),
                max(best[2], metric[2]),
            )
            print(
                "PROGRESS", metric, "best", best,
                "path_len", len(root_path), "path", root_path,
                flush=True,
            )

        candidates = [(canonical_walk, A_CONTROL)]
        if (24, 9) in by_position:
            candidates.append((by_position[(24, 9)], S_CONTROL))
        for position, walk in by_position.items():
            if position is None:
                continue
            candidates.append((walk, B_CONTROL))
            if position in D_BY_POSITION:
                label, control = D_BY_POSITION[position]
                candidates.append((walk, control))

        resulting_keys = set()
        for walk, control in candidates:
            probe = base.clone()
            apply(probe, root_path + walk)
            before = world_key(probe)
            step(probe, control)
            if probe.levels_completed > 5:
                print(
                    "WIN", root_path + walk + [control],
                    "states", len(seen), flush=True,
                )
                return
            after = world_key(probe)
            if after == before or after in resulting_keys:
                continue
            resulting_keys.add(after)
            queue.append(root_path + walk + [control])

        if len(seen) % 100 == 0:
            print("STATES", len(seen), "queue", len(queue), flush=True)
    print("DONE", len(seen), "queue", len(queue), "best", best, flush=True)


levels, path, error = A.run_program("dc22", run)
print("HARNESS", levels, len(path), error, flush=True)
