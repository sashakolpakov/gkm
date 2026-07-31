import importlib.util
import os
import sys

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

A_CONTROL = (6, 56, 8)
B_CONTROL = (6, 51, 25)
S_CONTROL = (6, 51, 48)
D_LEFT = (6, 46, 36)
D_CONTROLS = {
    "u": (1, (6, 50, 32), 2),
    "d": (2, (6, 50, 40), 1),
    "l": (3, D_LEFT, 4),
    "r": (4, (6, 54, 36), 3),
}
CONTROLS = (
    ("A", A_CONTROL),
    ("B", B_CONTROL),
    ("S", S_CONTROL),
    ("DU", (6, 50, 32)),
    ("DD", (6, 50, 40)),
    ("DL", D_LEFT),
    ("DR", (6, 54, 36)),
)

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
    blobs = connected_components(env.frame(), colors=(14,), min_area=1)
    candidates = [blob for blob in blobs if blob.bbox[1] < 40]
    if not candidates:
        return None
    blob = max(candidates, key=lambda item: item.area)
    return blob.bbox[0] // 2, blob.bbox[1] // 2, blob.area


def tile_map(env, rows=range(10, 22), cols=range(0, 20)):
    frame = env.frame()
    lines = []
    for row in rows:
        chars = []
        for col in cols:
            block = frame[2 * row:2 * row + 2, 2 * col:2 * col + 2]
            values = sorted({int(value) for value in block.flat})
            chars.append(f"{values[0]:X}" if len(values) == 1 else "*")
        lines.append(f"{row:02d} {''.join(chars)}")
    return lines


def describe(label, env):
    components = [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(env.frame(), min_area=4)
        if blob.bbox[1] < 40 and blob.color not in (2, 4)
    ]
    print(label, "avatar", avatar_tile(env), "components", components)
    for line in tile_map(env):
        print(label, line)


def configure_ring(env, labels):
    for label in labels:
        outward, control, inward = D_CONTROLS[label]
        apply(env, [outward, control, inward])


def run(env):
    solver.solve(env)
    level6 = env.clone()
    apply(env, SHIFTED_SELECTOR + REVERSE)
    root = env.clone()
    apply(root, [1, 1, 1, 4, 4])
    describe("BEFORE_B", root)

    opened = root.clone()
    before = opened.frame().copy()
    opened.step(*B_CONTROL)
    print("B_DELTA", frame_delta(before, opened.frame()))
    describe("AFTER_B", opened)

    no_b = root.clone()
    no_b.step(4)
    print("NO_B_RIGHT", avatar_tile(no_b))

    entered = opened.clone()
    entered.step(4)
    describe("ENTERED", entered)

    apply(entered, [1, 4, 4])
    describe("FAR_WALL", entered)
    before = entered.frame().copy()
    entered.step(*B_CONTROL)
    print("FAR_B_DELTA", frame_delta(before, entered.frame()))
    describe("FAR_AFTER_B", entered)

    queue = [(opened.clone(), [])]
    nodes = {}
    while queue:
        node, path = queue.pop(0)
        position = avatar_tile(node)
        if position in nodes:
            continue
        nodes[position] = (node, path)
        for direction in (1, 2, 3, 4):
            child = node.clone()
            child.step(direction)
            if avatar_tile(child) not in nodes:
                queue.append((child, path + [direction]))
    print("OPEN_REACH", sorted(nodes))
    for position, (node, path) in sorted(nodes.items()):
        effects = []
        for name, control in CONTROLS:
            child = node.clone()
            before = child.frame().copy()
            child.step(*control)
            delta = frame_delta(before[:62, :40], child.frame()[:62, :40])
            if delta["count"]:
                effects.append((name, delta["count"], delta["bbox"], avatar_tile(child)))
        if effects:
            print("OPEN_CONTROL", position, "path", path, "effects", effects)

    final_ring = level6.clone()
    apply(final_ring, HUB)
    configure_ring(final_ring, ["u", "r", "u", "u", "l", "l", "u", "u", "u"])
    apply(final_ring, [B_CONTROL, S_CONTROL, S_CONTROL, S_CONTROL, B_CONTROL])
    describe("FINAL_TOP", final_ring)

    action_set = (
        ("u", 1), ("d", 2), ("l", 3), ("r", 4),
        ("a", A_CONTROL), ("b", B_CONTROL),
    )
    queue = [(final_ring.clone(), [])]
    seen = {final_ring.frame()[:62, :40].tobytes()}
    positions = {avatar_tile(final_ring)}
    while queue and len(seen) < 3000:
        node, path = queue.pop(0)
        if len(path) >= 80:
            continue
        for label, action in action_set:
            child = node.clone()
            child.step(*action) if isinstance(action, tuple) else child.step(action)
            child_path = path + [label]
            if child.levels_completed > 5:
                print("FINAL_WIN", child_path, "states", len(seen), flush=True)
                return
            key = child.frame()[:62, :40].tobytes()
            if key in seen:
                continue
            seen.add(key)
            position = avatar_tile(child)
            if position not in positions:
                positions.add(position)
                if position is None or position[1] >= 6 or position[0] >= 7:
                    print("FINAL_FRONTIER", position, child_path, flush=True)
            queue.append((child, child_path))
    print("FINAL_DONE", len(seen), "positions", sorted(positions), flush=True)


A.run_program("dc22", run)
