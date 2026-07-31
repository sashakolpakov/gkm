import importlib.util
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import (
    arr,
    action_deltas,
    bounded_bfs,
    color_counts,
    level_goal,
    object_candidates,
    path_result,
)


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")

spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def player_distance(node, target):
    frame = arr(node.frame())
    ys, xs = __import__("numpy").where(
        ((frame == 9) | (frame == 10))
        & ~(
            (__import__("numpy").indices(frame.shape)[0] <= 14)
            & (__import__("numpy").indices(frame.shape)[1] >= 50)
        )
    )
    if not len(ys):
        return 999
    return min(
        abs(int(y) - target[0]) + abs(int(x) - target[1])
        for y, x in zip(ys, xs)
    )


def closest_route(env, target, max_states=1000, max_depth=60):
    start = env.clone()
    key = lambda node: arr(node.frame())[10:47, :60].tobytes()
    queue = deque([(start, [])])
    seen = {key(start)}
    best = (player_distance(start, target), [])
    largest_delta = (0, [], None)
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        distance = player_distance(node, target)
        if distance < best[0]:
            best = distance, path
        if len(path) >= max_depth:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            delta = __import__("perception").frame_delta(
                arr(node.frame())[10:47, :60],
                arr(child.frame())[10:47, :60],
            )
            if delta["count"] > largest_delta[0]:
                largest_delta = (
                    delta["count"], path + [action], delta["bbox"]
                )
            child_key = key(child)
            if child_key not in seen:
                seen.add(child_key)
                queue.append((child, path + [action]))
    return best, len(seen), largest_delta


GRID_POINTS = tuple(
    (x, y) for y in (50, 55, 60) for x in (25, 30, 35)
)


def choose_pattern(node, points):
    frame = arr(node.frame())
    wanted = set(points)
    actions = []
    for x, y in GRID_POINTS:
        selected = int(frame[y, x]) == 14
        if selected != ((x, y) in wanted):
            node.step(6, x, y)
            actions.append((6, x, y))
    return actions


def macro_bfs(env, patterns, max_states=2000, max_depth=25):
    key = lambda node: (
        arr(node.frame())[10:47, :60].tobytes(),
        tuple(int(arr(node.frame())[y, x]) for x, y in GRID_POINTS),
    )
    def reconstruct(macros):
        node = env.clone()
        primitives = []
        for label in macros:
            if isinstance(label, int):
                node.step(label)
                primitives.append(label)
            else:
                primitives.extend(choose_pattern(node, patterns[label]))
        return node, primitives

    start, _ = reconstruct([])
    queue = deque([[]])
    seen = {key(start)}
    while queue and len(seen) <= max_states:
        macros = queue.popleft()
        node, primitives = reconstruct(macros)
        if node.levels_completed > env.levels_completed:
            return macros, primitives, len(seen)
        if len(macros) >= max_depth or node.terminal():
            continue
        for operation in (1, 2, 3, 4, *patterns):
            child_macros = macros + [operation]
            child, child_primitives = reconstruct(child_macros)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            if child.levels_completed > env.levels_completed:
                return child_macros, child_primitives, len(seen)
            queue.append(child_macros)
    return None, None, len(seen)


def small_player_bbox(node):
    pieces = __import__("perception").connected_components(
        node.frame(), colors=(9, 10), min_area=1
    )
    moving = [b for b in pieces if b.area <= 8]
    if not moving:
        return None
    return (
        min(b.bbox[0] for b in moving),
        min(b.bbox[1] for b in moving),
        max(b.bbox[2] for b in moving),
        max(b.bbox[3] for b in moving),
    )


def probe(env):
    solver.solve(env)
    print("level", env.levels_completed, "actions", env.actions)
    print("colors", color_counts(env.frame()))
    symbols = "0123456789ABCDEF"
    frame = arr(env.frame())
    print("map_2x2_r10_47_c0_59")
    for r in range(10, 48, 2):
        line = []
        for c in range(0, 60, 2):
            vals, counts = __import__("numpy").unique(
                frame[r:r + 2, c:c + 2], return_counts=True
            )
            line.append(symbols[int(vals[counts.argmax()])])
        print("".join(line))
    print("objects")
    for obj in object_candidates(env.frame(), min_area=3):
        print(obj)
    print("key_deltas")
    for action, delta in action_deltas(env, actions=(1, 2, 3, 4)).items():
        print(action, {k: v for k, v in delta.items() if k != "samples"})
    print("click_deltas")
    click_points = [
        (25, 50), (30, 50), (35, 50),
        (25, 55), (30, 55), (35, 55),
        (25, 60), (30, 60), (35, 60),
        (36, 16), (52, 12), (16, 12), (4, 44), (12, 40), (30, 40),
    ]
    for x, y in click_points:
        clone = env.clone()
        before = arr(clone.frame()).copy()
        clone.step(6, x, y)
        delta = __import__("perception").frame_delta(before, clone.frame())
        print(
            (x, y), delta["count"], delta["bbox"],
            int(arr(clone.frame())[y, x]), clone.levels_completed,
        )
    patterns = {
        "15_diamond": ((30, 50), (25, 55), (35, 55), (30, 60)),
        "11_corner": ((25, 50), (30, 50), (30, 55)),
        "6_vertical": ((30, 50), (30, 55), (30, 60)),
    }
    macro_path, primitive_path, macro_states = macro_bfs(env, patterns)
    print("macro_goal", macro_path, primitive_path, "states", macro_states)
    persistence = env.clone()
    choose_pattern(persistence, patterns["6_vertical"])
    persistence.step(3)
    moved_6 = small_player_bbox(persistence)
    choose_pattern(persistence, patterns["15_diamond"])
    persistence.step(2)
    moved_15 = small_player_bbox(persistence)
    choose_pattern(persistence, patterns["6_vertical"])
    persistence.step(3)
    queued_6 = small_player_bbox(persistence)
    persistence.step(3)
    returned_6 = small_player_bbox(persistence)
    choose_pattern(persistence, patterns["15_diamond"])
    persistence.step(2)
    queued_15 = small_player_bbox(persistence)
    persistence.step(2)
    returned_15 = small_player_bbox(persistence)
    print(
        "persistence",
        "6", moved_6, queued_6, returned_6,
        "15", moved_15, queued_15, returned_15,
    )
    targets = {
        "15_diamond": {"switch12": (44, 4), "marker14": (40, 12)},
        "11_corner": {"goal": (12, 52)},
        "6_vertical": {"switch13": (12, 16), "marker15": (39, 29)},
    }
    print("pattern_then_key")
    for name, points in patterns.items():
        selected = env.clone()
        for x, y in points:
            selected.step(6, x, y)
        selected_pieces = [
            (b.color, b.bbox, b.area)
            for b in __import__("perception").connected_components(
                selected.frame(), colors=(4, 6, 9, 10, 11, 12, 13, 14, 15),
                min_area=4,
            )
            if 10 <= b.bbox[0] <= 47 and b.bbox[1] < 60
        ]
        print(name, "selected_pieces", selected_pieces)
        for action, delta in action_deltas(
            selected, actions=(1, 2, 3, 4)
        ).items():
            child = selected.clone()
            child.step(action)
            pieces = [
                (b.color, b.bbox, b.area)
                for b in __import__("perception").connected_components(
                    child.frame(), colors=(6, 9, 10, 11, 12, 13, 15),
                    min_area=4,
                )
            ]
            print(
                name, action, delta["count"], delta["bbox"],
                "level", child.levels_completed, "pieces", pieces,
            )
            if name == "15_diamond" and action == 1:
                print("15_up_samples", delta["samples"])
        path = bounded_bfs(
            selected,
            level_goal(selected.levels_completed),
            actions=(1, 2, 3, 4),
            key_fn=lambda node: arr(node.frame())[10:47, :60].tobytes(),
            max_states=2000,
            max_depth=60,
        )
        print(name, "movement_goal_path", path)
        for target_name, target in targets[name].items():
            (
                (closest_distance, closest_path),
                explored,
                largest_delta,
            ) = closest_route(
                selected, target, max_states=1000
            )
            print(
                name, target_name, "closest", closest_distance, closest_path,
                "states", explored, "largest_delta", largest_delta,
            )
            route = bounded_bfs(
                selected,
                lambda node, path, target=target: player_distance(
                    node, target
                ) <= 2,
                actions=(1, 2, 3, 4),
                key_fn=lambda node: arr(node.frame())[10:47, :60].tobytes(),
                max_states=1000,
                max_depth=60,
            )
            if route:
                reached = __import__("perception").replay(selected, route)
                print(
                    name, target_name, route,
                    "distance", player_distance(reached, target),
                    "level", reached.levels_completed,
                )
        selected_counts = color_counts(selected.frame())
        tracked = (6, 11, 12, 13, 15)
        interaction = bounded_bfs(
            selected,
            lambda node, path: any(
                color_counts(node.frame()).get(color, 0)
                != selected_counts.get(color, 0)
                for color in tracked
            ),
            actions=(1, 2, 3, 4),
            key_fn=lambda node: arr(node.frame())[10:47, :60].tobytes(),
            max_states=2000,
            max_depth=60,
        )
        print(name, "first_interaction", interaction)
        if interaction:
            result = path_result(selected, interaction)
            print(
                name, "interaction_counts",
                {color: result["colors"].get(color, 0) for color in tracked},
            )
    path = bounded_bfs(
        env,
        level_goal(env.levels_completed),
        actions=(1, 2, 3, 4),
        max_states=2000,
        max_depth=60,
    )
    print("movement_goal_path", path)
    if path:
        print("movement_goal_result", path_result(env, path)["levels_completed"])


A.run_program("sc25", probe)
