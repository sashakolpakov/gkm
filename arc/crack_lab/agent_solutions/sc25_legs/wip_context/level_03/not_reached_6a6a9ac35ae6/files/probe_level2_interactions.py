import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from legs import select_grid_cells_of_color
from perception import bounded_bfs, color_counts, frame_delta, level_goal


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def lower_pose(frame):
    board = np.asarray(frame)
    mask = ((board == 9) | (board == 10))
    mask[:15] = False
    points = np.argwhere(mask)
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    chars = {9: "a", 10: "b"}
    pattern = "/".join(
        "".join(chars.get(int(value), ".") for value in row)
        for row in board[r0:r1 + 1, c0:c1 + 1]
    )
    return (int(r0), int(c0), int(r1), int(c1), pattern)


def probe(env):
    solver.players.play_level_1(env)
    root = env.clone()

    # Enumerate movement poses while ignoring the countdown bar.
    nodes = [root]
    paths = [[]]
    keys = {np.asarray(root.frame())[15:39, 27:39].tobytes(): 0}
    edges = {}
    index = 0
    while index < len(nodes):
        node = nodes[index]
        edges[index] = {}
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = np.asarray(child.frame())[15:39, 27:39].tobytes()
            if key not in keys:
                keys[key] = len(nodes)
                nodes.append(child)
                paths.append(paths[index] + [action])
            edges[index][action] = keys[key]
        index += 1
    for node_id, node in enumerate(nodes):
        print("pose", node_id, "via", paths[node_id], lower_pose(node.frame()),
              "edges", edges[node_id])

    # Sample every two pixels: enough to cross any visible one-pixel feature.
    direct = {}
    for action in (1, 2, 3, 4):
        child = root.clone()
        child.step(action)
        direct[action] = np.asarray(child.frame()).tobytes()
    effects = []
    for y in range(0, 64, 2):
        for x in range(0, 64, 2):
            clicked = root.clone()
            clicked.step(6, x, y)
            if np.asarray(clicked.frame()).tobytes() != np.asarray(root.frame()).tobytes():
                effects.append((x, y, "direct"))
                continue
            for action in (1, 2, 3, 4):
                child = clicked.clone()
                child.step(action)
                if (child.levels_completed != root.levels_completed
                        or np.asarray(child.frame()).tobytes() != direct[action]):
                    effects.append((x, y, action))
    print("sampled_click_effects", effects)

    xs, ys = (25, 30, 35), (50, 55, 60)
    print("grid_before", [
        [int(root.frame()[y][x]) for x in xs]
        for y in ys
    ])
    for y in ys:
        for x in xs:
            child = root.clone()
            child.step(6, x, y)
            delta = frame_delta(root.frame(), child.frame())
            changes = {}
            before = np.asarray(root.frame())
            after = np.asarray(child.frame())
            changed = before != after
            for old, new in zip(before[changed], after[changed]):
                pair = (int(old), int(new))
                changes[pair] = changes.get(pair, 0) + 1
            print("cell", (x, y), "center", int(before[y, x]), "delta",
                  (delta["count"], delta["bbox"]), "transitions", sorted(changes.items()))

    for color in (0, 2, 3):
        child = root.clone()
        select_grid_cells_of_color(child, xs, ys, color)
        print("select_color", color, "grid", [
            [int(child.frame()[y][x]) for x in xs]
            for y in ys
        ], "counts", color_counts(child.frame()))
        path = bounded_bfs(
            child,
            level_goal(root.levels_completed),
            actions=(1, 2, 3, 4),
            max_states=5000,
            max_depth=32,
        )
        print("select_color_bfs", color, path)


if __name__ == "__main__":
    A.run_program("sc25", probe)
