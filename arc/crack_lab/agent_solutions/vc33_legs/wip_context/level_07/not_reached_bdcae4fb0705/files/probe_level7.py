"""Compact clean-room probes for vc33 level 7 via the public env surface."""
import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, color_counts, connected_components, frame_delta
from legs import (
    align_marker_pair_with_pressure_controls,
    cross_pressure_gates_then_align_height,
    relay_height_between_adjacent_reservoirs,
)


with open("checkpoint.json") as f:
    PREFIX = json.load(f)["final_path"]


def summary(blob):
    return (
        blob.color,
        blob.area,
        blob.bbox,
        tuple(round(v, 1) for v in blob.centroid),
    )


def probe(env):
    for action in PREFIX:
        env.step(*action)
    print("level", env.levels_completed, "actions", env.actions)
    print("counts", color_counts(env.frame()))
    blobs = connected_components(env.frame(), min_area=2)
    print("blobs", [summary(blob) for blob in blobs])

    base = arr(env.frame())
    groups = {}
    for y in range(2, 64, 4):
        for x in range(2, 64, 4):
            child = env.clone()
            child.step(6, x, y)
            delta = frame_delta(base, child.frame())
            if not delta["count"]:
                continue
            key = (arr(child.frame()).tobytes(), child.levels_completed)
            group = groups.setdefault(
                key,
                {
                    "points": [],
                    "delta": (delta["count"], delta["bbox"]),
                    "counts": color_counts(child.frame()),
                    "level": child.levels_completed,
                },
            )
            group["points"].append((x, y))
    print(
        "effects",
        [
            {
                "n": len(group["points"]),
                "reps": group["points"][:6],
                "delta": group["delta"],
                "counts": group["counts"],
                "level": group["level"],
            }
            for group in groups.values()
        ],
    )

    exact = []
    for blob in connected_components(env.frame(), colors=(9,), min_area=2):
        y, x = (int(v) for v in blob.centroid)
        child = env.clone()
        child.step(6, x, y)
        delta = frame_delta(base, child.frame())
        exact.append(((x, y), delta["count"], delta["bbox"], color_counts(child.frame())))
    print("exact_controls", exact)

    child = env.clone()
    relay_height_between_adjacent_reservoirs(child)
    print(
        "relay",
        child.levels_completed,
        color_counts(child.frame()),
        [
            summary(blob)
            for blob in connected_components(
                child.frame(), colors=(1, 9, 11, 14, 15), min_area=2
            )
        ],
    )

    marker_colors = (11, 14, 15)

    def gaps(node):
        result = {}
        for color in marker_colors:
            pair = connected_components(node.frame(), colors=(color,), min_area=1)
            if len(pair) == 2:
                result[color] = abs(pair[0].centroid[0] - pair[1].centroid[0])
        return result

    controls = [
        (6, int(blob.centroid[1]), int(blob.centroid[0]))
        for blob in connected_components(env.frame(), colors=(9,), min_area=2)
    ]
    initial = sum(gaps(env).values())
    queue = deque([(env.clone(), [])])
    seen = {arr(env.frame())[1:].tobytes()}
    found = None
    best = (initial, [])
    while queue and len(seen) <= 600:
        node, path = queue.popleft()
        if len(path) >= 10:
            continue
        for action in controls:
            child = node.clone()
            child.step(*action)
            child_path = path + [action]
            value = sum(gaps(child).values())
            if value < best[0]:
                best = (value, child_path)
            if child.levels_completed > env.levels_completed or value < initial:
                found = (child.levels_completed, gaps(child), child_path)
                queue.clear()
                break
            key = arr(child.frame())[1:].tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path))
    print("gap_search", initial, len(seen), best, found)

    child = env.clone()
    greedy_path = []
    trace = [gaps(child)]
    for _ in range(80):
        current = sum(gaps(child).values())
        choice = None
        for action in controls:
            candidate = child.clone()
            candidate.step(*action)
            value = sum(gaps(candidate).values())
            if candidate.levels_completed > env.levels_completed:
                choice = (value, action, candidate)
                break
            if value < current and (choice is None or value < choice[0]):
                choice = (value, action, candidate)
        if choice is None:
            break
        _, action, child = choice
        greedy_path.append(action)
        trace.append(gaps(child))
        if child.levels_completed > env.levels_completed:
            break
    print("greedy", child.levels_completed, greedy_path, trace)
    print(
        "greedy_state",
        [
            summary(blob)
            for blob in connected_components(
                child.frame(), colors=(0, 1, 3, 11, 12, 13, 14, 15), min_area=2
            )
        ],
    )
    child = env.clone()
    cross_pressure_gates_then_align_height(child, marker_color=11)
    print(
        "cross11",
        child.levels_completed,
        gaps(child),
        [
            summary(blob)
            for blob in connected_components(
                child.frame(), colors=(1, 9, 11, 12, 13, 14, 15), min_area=2
            )
        ],
    )
    align_marker_pair_with_pressure_controls(child, marker_color=11)
    print("cross11_align", child.levels_completed, gaps(child))
    return

    child = env.clone()
    staged_path = []
    staged_trace = [gaps(child)]
    for _ in range(40):
        baseline = sum(gaps(child).values())
        queue = deque([(child.clone(), [])])
        seen = {arr(child.frame())[1:].tobytes()}
        stage = None
        while queue and len(seen) <= 1200:
            node, path = queue.popleft()
            if len(path) >= 12:
                continue
            options = [
                (6, int(blob.centroid[1]), int(blob.centroid[0]))
                for blob in connected_components(
                    node.frame(), colors=(9, 12, 13, 14, 15), min_area=2
                )
            ]
            for action in options:
                candidate = node.clone()
                candidate.step(*action)
                candidate_path = path + [action]
                value = sum(gaps(candidate).values())
                if (
                    candidate.levels_completed > env.levels_completed
                    or value < baseline
                ):
                    stage = candidate_path
                    queue.clear()
                    break
                key = arr(candidate.frame())[1:].tobytes()
                if key not in seen:
                    seen.add(key)
                    queue.append((candidate, candidate_path))
        if stage is None:
            print("staged_stall", baseline, len(seen))
            break
        for action in stage:
            child.step(*action)
        staged_path.extend(stage)
        staged_trace.append(gaps(child))
        if child.levels_completed > env.levels_completed:
            break
    print("staged", child.levels_completed, len(staged_path), staged_path, staged_trace)


arena.run_program("vc33", probe)
