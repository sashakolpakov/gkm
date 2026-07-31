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
D_LEFT = (6, 46, 36)
REMOTE = [
    3, 3, 3, 3, 3,
    A_CONTROL, A_CONTROL, A_CONTROL, A_CONTROL,
    2, 2, 3, 3, 3, 2, 3, A_CONTROL, 1, A_CONTROL, 1, 1,
    B_CONTROL,
] + [1] * 17 + [3] + [2] * 11
SHIFTED_SELECTOR = (
    REMOTE
    + [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
    + [
        S_CONTROL, S_CONTROL, S_CONTROL, 3, B_CONTROL,
        3, D_LEFT, 4,
        3, D_LEFT, 4,
        B_CONTROL,
    ]
)
REVERSE = [
    4,
    2, 2, 2, 3, 2, 3, 3, 2, 3, 3, 1, A_CONTROL, 1,
] + [1] * 7 + [3]


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar_tile(env):
    blobs = connected_components(env.frame(), colors=(14,), min_area=2)
    if not blobs:
        return None
    blob = max(blobs, key=lambda item: item.area)
    return blob.bbox[0] // 2, blob.bbox[1] // 2


def solid_tiles(env):
    frame = env.frame()
    out = []
    for row in range(31):
        for col in range(32):
            block = frame[2 * row:2 * row + 2, 2 * col:2 * col + 2]
            if (block == block[0, 0]).all() and int(block[0, 0]) not in (0, 2, 4, 5):
                out.append((int(block[0, 0]), row, col))
    return out


def ring_bounds(env):
    blobs = [
        blob for blob in connected_components(env.frame(), colors=(8,), min_area=8)
        if blob.bbox[1] < 40
    ]
    if not blobs:
        return None
    blob = max(blobs, key=lambda item: item.area)
    return blob.bbox, blob.area


def run(env):
    solver.solve(env)
    apply(env, SHIFTED_SELECTOR)
    print("SELECTOR", avatar_tile(env))
    for index, action in enumerate(REVERSE, 1):
        env.step(*action) if isinstance(action, tuple) else env.step(action)
        if index in (1, 4, 13, 20, 21):
            print("STEP", index, avatar_tile(env))
    queue = deque([(env.clone(), [])])
    paths = {avatar_tile(env): []}
    while queue and len(paths) < 180:
        node, path = queue.popleft()
        for direction in (1, 2, 3, 4):
            child = node.clone()
            child.step(direction)
            position = avatar_tile(child)
            if position not in paths:
                paths[position] = path + [direction]
                if position is None:
                    print("OCCLUDED", paths[position], flush=True)
                queue.append((child, path + [direction]))
    ordered_positions = sorted(position for position in paths if position is not None)
    frontiers = [
        (position, paths[position])
        for position in ordered_positions
        if position[1] >= 10 or position[0] <= 7
    ]
    print(
        "REACH", len(paths),
        "bounds", (min(ordered_positions), max(ordered_positions)),
        "frontiers", frontiers,
    )
    hidden = env.clone()
    apply(hidden, [1, 1, 4, S_CONTROL, S_CONTROL, B_CONTROL])
    print(
        "HIDDEN_PHASE1", avatar_tile(hidden),
        hidden.levels_completed,
        [
            (int(row), int(col))
            for row, col in zip(*((hidden.frame() == 14).nonzero()))
        ],
        flush=True,
    )
    handoff = env.clone()
    apply(handoff, [1, 1, 1, 4, 4, B_CONTROL, 4, 4, 4])
    cells = [
        (int(row), int(col))
        for row, col in zip(*((handoff.frame() == 14).nonzero()))
    ]
    print("HANDOFF_ROOT", cells, "level", handoff.levels_completed, flush=True)
    for direction in (1, 2, 3, 4):
        child = handoff.clone()
        child.step(direction)
        child_cells = [
            (int(row), int(col))
            for row, col in zip(*((child.frame() == 14).nonzero()))
        ]
        print(
            "HANDOFF_NEXT", direction, child_cells,
            "level", child.levels_completed, flush=True,
        )
    route = env.clone()
    ring_route = [
        1, 1, 1, 4, 4, B_CONTROL, 4,
        1, 4, 4, B_CONTROL, 4, 2, 4,
    ]
    trace = []
    for action in ring_route:
        route.step(*action) if isinstance(action, tuple) else route.step(action)
        trace.append((avatar_tile(route), route.levels_completed))
    print("RING_ROUTE", trace, flush=True)
    inside = env.clone()
    apply(
        inside,
        [1, 1, 1, 4, 4, B_CONTROL, 4, 1, 4, 4],
    )
    for name, control in (
        ("u", (6, 50, 32)),
        ("d", (6, 50, 40)),
        ("l", D_LEFT),
        ("r", (6, 54, 36)),
    ):
        child = inside.clone()
        before_position = avatar_tile(child)
        child.step(*control)
        print(
            "INSIDE_CONTROL", name, before_position, avatar_tile(child),
            ring_bounds(inside), ring_bounds(child),
            child.levels_completed, flush=True,
        )
    ring_teleport = inside.clone()
    apply(ring_teleport, [S_CONTROL, S_CONTROL, B_CONTROL])
    print(
        "INSIDE_PHASE1", avatar_tile(ring_teleport),
        ring_teleport.levels_completed, flush=True,
    )
    ring_entry = [1, 1, 1, 4, 4, B_CONTROL, 4]
    for interior_path in (
        [1, 4],
        [1, 4, 4],
        [1, 4, 2],
        [1, 4, 4, 2],
    ):
        interior = env.clone()
        apply(interior, ring_entry + interior_path)
        control_effects = []
        for name, control in (
            ("u", (6, 50, 32)),
            ("d", (6, 50, 40)),
            ("l", D_LEFT),
            ("r", (6, 54, 36)),
        ):
            moved = interior.clone()
            before_ring = ring_bounds(moved)
            moved.step(*control)
            after_ring = ring_bounds(moved)
            if before_ring != after_ring or moved.levels_completed > 5:
                control_effects.append((
                    name, before_ring, after_ring,
                    avatar_tile(moved), moved.levels_completed,
                ))
        occupant = env.clone()
        apply(
            occupant,
            ring_entry + interior_path + [S_CONTROL, S_CONTROL, B_CONTROL],
        )
        print(
            "INTERIOR_PHASE1", interior_path,
            avatar_tile(occupant), occupant.levels_completed,
            "controls", control_effects,
            flush=True,
        )
    for selector_phase in range(4):
        operated = inside.clone()
        for _ in range((selector_phase - 3) % 4):
            operated.step(*S_CONTROL)
        operated.step(*B_CONTROL)
        effects = []
        for name, control in (
            ("u", (6, 50, 32)),
            ("d", (6, 50, 40)),
            ("l", D_LEFT),
            ("r", (6, 54, 36)),
        ):
            child = operated.clone()
            before_ring = ring_bounds(child)
            child.step(*control)
            after_ring = ring_bounds(child)
            if after_ring != before_ring or child.levels_completed > 5:
                effects.append((
                    name, before_ring, after_ring,
                    avatar_tile(child), child.levels_completed,
                ))
        print("COOP_PHASE", selector_phase, effects, flush=True)
    actions = [
        ("u", 1), ("d", 2), ("l", 3), ("r", 4), ("b", B_CONTROL),
    ]
    state_queue = deque([(env.clone(), [])])
    seen = {env.frame()[:63].tobytes()}
    positions = {avatar_tile(env)}
    print("START_B", env.levels_completed, bool(state_queue), len(seen), flush=True)
    while state_queue and len(seen) < 800:
        node, path = state_queue.popleft()
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
                if position is None:
                    print(
                        "HANDOFF", child_path, solid_tiles(child),
                        "level", child.levels_completed, flush=True,
                    )
                elif position[1] >= 10 or position[0] <= 7:
                    print("B_FRONTIER", position, child_path, flush=True)
            state_queue.append((child, child_path))
    print(
        "B_DONE", len(seen), "queue", len(state_queue),
        "positions", len(positions), flush=True,
    )


levels, path, error = A.run_program("dc22", run)
print("HARNESS", levels, len(path), error, flush=True)
