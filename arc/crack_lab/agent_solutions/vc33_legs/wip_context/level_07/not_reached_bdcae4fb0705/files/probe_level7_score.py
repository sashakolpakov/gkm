"""Verify a general dense score for coupled marked pressure relays."""
import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, connected_components


with open("checkpoint.json") as f:
    PREFIX = json.load(f)["final_path"]


def probe(env):
    for action in PREFIX:
        env.step(*action)
    start_level = env.levels_completed
    marker_colors = (11, 14, 15)

    def pairs(node):
        result = {}
        for color in marker_colors:
            markers = [
                blob
                for blob in connected_components(
                    node.frame(), colors=(color,), min_area=1
                )
                if blob.area <= 5
            ]
            if len(markers) == 2:
                moving = max(markers, key=lambda blob: blob.area)
                fixed = min(markers, key=lambda blob: blob.area)
                result[color] = (
                    abs(moving.centroid[1] - fixed.centroid[1]),
                    abs(moving.centroid[0] - fixed.centroid[0]),
                )
        return result

    def score(node):
        values = pairs(node)
        uncrossed = [value for value in values.values() if value[0] > 12]
        crossed = [value for value in values.values() if value[0] <= 12]
        return (
            len(values),
            len(uncrossed),
            sum(value[0] - 12 for value in uncrossed),
            sum(value[1] for value in crossed),
        )

    path = []
    trace = [(score(env), pairs(env))]
    for _ in range(48):
        baseline = score(env)
        queue = deque([(env.clone(), [])])
        seen = {arr(env.frame())[1:].tobytes()}
        stage = None
        while queue and len(seen) <= 1600:
            node, prefix = queue.popleft()
            if len(prefix) >= 20:
                continue
            actions = [
                (6, int(blob.centroid[1]), int(blob.centroid[0]))
                for blob in connected_components(
                    node.frame(), colors=(9,), min_area=2
                )
            ]
            actions.extend(
                (6, int(blob.centroid[1]), int(blob.centroid[0]))
                for blob in connected_components(
                    node.frame(), colors=(12, 13, 14, 15), min_area=8
                )
            )
            for action in actions:
                child = node.clone()
                child.step(*action)
                candidate = prefix + [action]
                if child.levels_completed > start_level or score(child) < baseline:
                    stage = candidate
                    queue.clear()
                    break
                key = arr(child.frame())[1:].tobytes()
                if key not in seen:
                    seen.add(key)
                    queue.append((child, candidate))
        if stage is None:
            print("stall", baseline, len(seen))
            break
        for action in stage:
            env.step(*action)
        path.extend(stage)
        trace.append((score(env), pairs(env)))
        if env.levels_completed > start_level:
            break
    print("result", env.levels_completed, len(path), path)
    print("trace", trace)


arena.run_program("vc33", probe)
