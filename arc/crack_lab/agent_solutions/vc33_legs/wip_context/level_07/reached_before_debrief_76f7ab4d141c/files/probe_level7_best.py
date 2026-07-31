"""Bounded reward search over level 7's visible pressure controls."""
import heapq
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, connected_components
from legs import align_marker_pair_with_pressure_controls


with open("checkpoint.json") as f:
    PREFIX = json.load(f)["final_path"]


def probe(env):
    for action in PREFIX:
        env.step(*action)
    start_level = env.levels_completed
    marker_colors = (11, 14, 15)

    for action in ((6, 24, 8), (6, 42, 8), (6, 40, 19)):
        env.step(*action)
    align_marker_pair_with_pressure_controls(
        env, marker_color=15, max_stages=24, max_states=1200, max_depth=18
    )
    for _ in range(3):
        env.step(6, 38, 32)

    def gaps(node):
        values = []
        for color in marker_colors:
            blobs = [
                blob
                for blob in connected_components(
                    node.frame(), colors=(color,), min_area=1
                )
                if blob.area <= 5
            ]
            if len(blobs) == 2:
                moving = max(blobs, key=lambda blob: blob.area)
                fixed = min(blobs, key=lambda blob: blob.area)
                values.append(
                    (
                        color,
                        abs(moving.centroid[0] - fixed.centroid[0]),
                        abs(moving.centroid[1] - fixed.centroid[1]),
                    )
                )
        return tuple(values)

    def heuristic(node):
        return sum(vertical + horizontal for _, vertical, horizontal in gaps(node))

    def actions(node):
        options = [
            (6, int(blob.centroid[1]), int(blob.centroid[0]))
            for blob in connected_components(node.frame(), colors=(9,), min_area=2)
        ]
        options.extend(
            (6, int(blob.centroid[1]), int(blob.centroid[0]))
            for blob in connected_components(
                node.frame(), colors=(12, 13, 14, 15), min_area=8
            )
        )
        return options

    root = env.clone()
    counter = 0
    heap = [(heuristic(root), 0, counter, root, [])]
    seen = {arr(root.frame())[1:].tobytes()}
    best = heuristic(root)
    solution = None
    while heap and len(seen) <= 20000:
        _, depth, _, node, path = heapq.heappop(heap)
        if depth >= 80 or node.terminal():
            continue
        for action in actions(node):
            child = node.clone()
            child.step(*action)
            key = arr(child.frame())[1:].tobytes()
            if key in seen:
                continue
            seen.add(key)
            child_path = path + [action]
            if child.levels_completed > start_level:
                solution = child_path
                heap.clear()
                break
            value = heuristic(child)
            if value < best:
                best = value
                print("best", len(seen), depth + 1, value, gaps(child))
            counter += 1
            heapq.heappush(
                heap, (value + 0.15 * (depth + 1), depth + 1, counter, child, child_path)
            )
    print("result", len(seen), best, solution)


arena.run_program("vc33", probe)
