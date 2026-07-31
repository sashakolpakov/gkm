"""Compact clean-room probes for vc33 level 6 via the public env surface."""
import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, color_counts, connected_components, frame_delta
from legs import cross_pressure_gates_then_align_height


with open("checkpoint.json") as f:
    PREFIX = json.load(f)["final_path"]


def summarize(blob):
    return (
        blob.color,
        blob.area,
        blob.bbox,
        (round(blob.centroid[0], 1), round(blob.centroid[1], 1)),
    )


def probe(env):
    for action in PREFIX:
        env.step(*action)
    print("level", env.levels_completed, "actions", env.actions)
    print("counts", color_counts(env.frame()))
    blobs = connected_components(env.frame(), min_area=2)
    print("blobs", [summarize(b) for b in blobs])

    base = arr(env.frame())
    effective = []
    for blob in blobs:
        y, x = (int(round(v)) for v in blob.centroid)
        child = env.clone()
        child.step(6, x, y)
        delta = frame_delta(base, child.frame())
        if delta["count"]:
            effective.append(
                (
                    (blob.color, blob.area, x, y),
                    delta,
                    child.levels_completed,
                    color_counts(child.frame()),
                )
            )
    print("effective", effective)

    groups = {}
    for y in range(2, 64, 4):
        for x in range(2, 64, 4):
            child = env.clone()
            child.step(6, x, y)
            delta = frame_delta(base, child.frame())
            if not delta["count"]:
                continue
            key = (arr(child.frame()).tobytes(), child.levels_completed)
            entry = groups.setdefault(
                key,
                {
                    "points": [],
                    "delta": (delta["count"], delta["bbox"]),
                    "counts": color_counts(child.frame()),
                },
            )
            entry["points"].append((x, y))
    print(
        "grid_effects",
        [
            {
                "n_points": len(g["points"]),
                "reps": g["points"][:8],
                "delta": g["delta"],
                "counts": g["counts"],
            }
            for g in groups.values()
        ],
    )


def features(env):
    keep = (1, 4, 9, 11, 12, 13, 14, 15)
    return [
        summarize(b)
        for b in connected_components(env.frame(), colors=keep, min_area=2)
    ]


def probe_existing_leg(env):
    for action in PREFIX:
        env.step(*action)
    print("before_leg", features(env))
    cross_pressure_gates_then_align_height(env)
    print("after_leg", env.levels_completed, features(env))
    start_level = env.levels_completed
    queue = deque([(env.clone(), [])])
    seen = {arr(env.frame())[1:].tobytes()}
    answer = None
    while queue and len(seen) <= 3000:
        node, path = queue.popleft()
        if len(path) >= 28:
            continue
        options = [
            (6, int(b.centroid[1]), int(b.centroid[0]))
            for b in connected_components(
                node.frame(), colors=(9, 12, 13, 14, 15), min_area=2
            )
        ]
        for action in options:
            child = node.clone()
            child.step(*action)
            child_path = path + [action]
            if child.levels_completed > start_level:
                answer = child_path
                queue.clear()
                break
            key = arr(child.frame())[1:].tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path))
    print("finish_bfs", len(seen), answer)


arena.run_program("vc33", probe_existing_leg)
