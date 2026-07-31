import importlib.util
import sys
from collections import Counter, deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import numpy as np

from perception import (
    action_deltas,
    bounded_bfs,
    color_counts,
    frame_delta,
    level_goal,
    object_candidates,
)


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def compact_objects(frame):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in object_candidates(frame, min_area=4)
    ]


def probe(env):
    solver.players.play_level_1(env)
    print("level", env.levels_completed, "actions", env.actions)
    print("colors", color_counts(env.frame()))
    print("objects", compact_objects(env.frame()))
    print(
        "keys",
        {
            action: (delta["count"], delta["bbox"])
            for action, delta in action_deltas(env, (1, 2, 3, 4, 6)).items()
        },
    )
    before = np.asarray(env.frame())
    for action in (1, 2, 3, 4):
        clone = env.clone()
        clone.step(action)
        after = np.asarray(clone.frame())
        changed = before != after
        transitions = Counter(
            (int(a), int(b)) for a, b in zip(before[changed], after[changed])
        )
        tracked = [
            item for item in compact_objects(after)
            if item[0] in (4, 9, 10, 11)
        ]
        print("action", action, "transitions", sorted(transitions.items()), "tracked", tracked)
    base = env.frame()
    coords = [
        (32, 12),
        (31, 28),
        (35, 28),
        (32, 32),
        (32, 36),
        (32, 38),
        (13, 52),
        (16, 55),
    ]
    clicks = {}
    for x, y in coords:
        clone = env.clone()
        clone.step(6, x, y)
        delta = frame_delta(base, clone.frame())
        clicks[(x, y)] = (delta["count"], delta["bbox"])
    print("clicks", clicks)
    direct = {}
    for action in (1, 2, 3, 4):
        clone = env.clone()
        clone.step(action)
        direct[action] = np.asarray(clone.frame()).copy()
    selection_points = {
        "target9": (30, 10),
        "target10": (32, 12),
        "barrier4": (30, 27),
        "mover9": (32, 35),
        "mover10": (32, 37),
        "old11": (12, 51),
    }
    for name, (x, y) in selection_points.items():
        effects = {}
        for action in (1, 2, 3, 4):
            clone = env.clone()
            clone.step(6, x, y)
            clone.step(action)
            delta = frame_delta(direct[action], clone.frame())
            effects[action] = (delta["count"], delta["bbox"])
        print("select_then_key", name, effects)
    chars = {2: ".", 4: "#", 9: "a", 10: "b"}
    frame = np.asarray(env.frame())
    print("board")
    print("target")
    for row in frame[10:15, 30:36]:
        print("".join(chars.get(int(value), " ") for value in row))
    for row in frame[15:39, 27:39]:
        print("".join(chars.get(int(value), " ") for value in row))
    path = bounded_bfs(
        env,
        level_goal(env.levels_completed),
        actions=(1, 2, 3, 4),
        max_states=5000,
        max_depth=32,
    )
    print("bfs", path)
    queue = deque([(env.clone(), [])])
    seen = {np.asarray(env.frame())[15:39, 27:39].tobytes()}
    best = (99, [])
    while queue and len(seen) < 1000:
        node, path_here = queue.popleft()
        board_here = np.asarray(node.frame())
        points = np.argwhere(
            ((board_here == 9) | (board_here == 10))
            & (np.indices(board_here.shape)[0] >= 15)
        )
        if len(points) and int(points[:, 0].min()) < best[0]:
            best = (int(points[:, 0].min()), path_here)
        if len(path_here) >= 31 or node.terminal():
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = np.asarray(child.frame())[15:39, 27:39].tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((child, path_here + [action]))
    print("dense_search", "states", len(seen), "min_row", best[0], "path", best[1])
    state_nodes = {}
    start_key = np.asarray(env.frame())[15:39, 27:39].tobytes()
    state_nodes[start_key] = (env.clone(), [])
    graph_queue = deque([start_key])
    graph = {}
    while graph_queue:
        key = graph_queue.popleft()
        node, node_path = state_nodes[key]
        edges = {}
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_key = np.asarray(child.frame())[15:39, 27:39].tobytes()
            if child_key not in state_nodes:
                state_nodes[child_key] = (child, node_path + [action])
                graph_queue.append(child_key)
            edges[action] = child_key
        graph[key] = edges
    keys = list(state_nodes)
    ids = {key: index for index, key in enumerate(keys)}
    edge_ids = {
        ids[key]: {action: ids[target] for action, target in edges.items()}
        for key, edges in graph.items()
    }
    best_walk = (1, [])
    walk_seen = {}

    def cover(node_id, mask, path_here):
        nonlocal best_walk
        score = mask.bit_count()
        if score > best_walk[0]:
            best_walk = (score, path_here)
        if len(path_here) >= 31 or score == len(keys):
            return
        memo_key = (node_id, mask)
        if walk_seen.get(memo_key, 99) <= len(path_here):
            return
        walk_seen[memo_key] = len(path_here)
        for action, target_id in edge_ids[node_id].items():
            cover(target_id, mask | (1 << target_id), path_here + [action])

    cover(ids[start_key], 1 << ids[start_key], [])
    coverage_clone = env.clone()
    for action in best_walk[1]:
        if coverage_clone.terminal():
            break
        coverage_clone.step(action)
    print(
        "pose_coverage",
        best_walk,
        "level",
        coverage_clone.levels_completed,
        "terminal",
        coverage_clone.terminal(),
    )
    sequences = {
        "up1": [1],
        "up2": [1, 1],
        "up3": [1, 1, 1],
        "up4": [1, 1, 1, 1],
        "up8": [1] * 8,
        "left_up8": [3] + [1] * 8,
        "right_up8": [4] + [1] * 8,
        "zig_left": [3, 1, 3, 1, 4, 1, 4, 1],
        "zig_right": [4, 1, 4, 1, 3, 1, 3, 1],
        "vertical32": [1, 2] * 16,
        "horizontal32": [3, 4] * 16,
        "square32": [1, 3, 2, 4] * 8,
        "up32": [1] * 32,
        "down32": [2] * 32,
        "left32": [3] * 32,
        "right32": [4] * 32,
    }
    for name, actions in sequences.items():
        clone = env.clone()
        last_before = clone.frame()
        for action in actions:
            if clone.terminal():
                break
            last_before = clone.frame()
            clone.step(action)
        moving = [
            item for item in compact_objects(clone.frame())
            if item[0] in (9, 10) and item[1][0] >= 15
        ]
        print(
            "sequence",
            name,
            "level",
            clone.levels_completed,
            "terminal",
            clone.terminal(),
            "bar14",
            color_counts(clone.frame()).get(14, 0),
            "moving",
            moving,
        )
        if clone.terminal():
            print(
                "terminal_delta",
                name,
                (frame_delta(last_before, clone.frame())["count"],
                 frame_delta(last_before, clone.frame())["bbox"]),
                "colors",
                color_counts(clone.frame()),
            )


A.run_program("sc25", probe)
