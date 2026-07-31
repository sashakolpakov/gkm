"""Bounded symbolic search for the first level-7 gate traversal."""
import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, connected_components
from legs import align_marker_pair_with_pressure_controls


with open("checkpoint.json") as f:
    PREFIX = json.load(f)["final_path"]


def probe(env):
    for action in PREFIX:
        env.step(*action)
    base_level = env.levels_completed
    colors = (11, 14, 15)

    def gaps(node):
        result = {}
        for color in colors:
            pair = [
                blob
                for blob in connected_components(
                    node.frame(), colors=(color,), min_area=1
                )
                if blob.area <= 5
            ]
            if len(pair) == 2:
                moving = max(pair, key=lambda blob: blob.area)
                fixed = min(pair, key=lambda blob: blob.area)
                result[color] = (
                    abs(moving.centroid[1] - fixed.centroid[1]),
                    abs(moving.centroid[0] - fixed.centroid[0]),
                )
        return result

    initial = gaps(env)
    initial_horizontal = sum(value[0] for value in initial.values())
    queue = deque([(env.clone(), [])])
    seen = {arr(env.frame())[1:].tobytes()}
    found = None
    best = (initial_horizontal, None)
    while queue and len(seen) <= 6000:
        node, path = queue.popleft()
        if len(path) >= 32:
            continue
        options = [
            (6, int(blob.centroid[1]), int(blob.centroid[0]))
            for blob in connected_components(
                node.frame(), colors=(9, 12, 13, 14, 15), min_area=2
            )
        ]
        for action in options:
            child = node.clone()
            child.step(*action)
            child_path = path + [action]
            state_gaps = gaps(child)
            horizontal = sum(value[0] for value in state_gaps.values())
            if horizontal < best[0]:
                best = (horizontal, child_path)
            if child.levels_completed > base_level or horizontal < initial_horizontal:
                found = (child.levels_completed, state_gaps, child_path)
                queue.clear()
                break
            key = arr(child.frame())[1:].tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path))
    print("search", initial, len(seen), best, found)

    node = env.clone()
    full_path = []
    trace = [gaps(node)]
    for action in found[2]:
        node.step(*action)
    full_path.extend(found[2])
    trace.append(gaps(node))
    for _ in range(0):
        baseline = sum(value[0] for value in gaps(node).values())
        queue = deque([(node.clone(), [])])
        seen = {arr(node.frame())[1:].tobytes()}
        stage = None
        while queue and len(seen) <= 3000:
            state, path = queue.popleft()
            if len(path) >= 28:
                continue
            options = [
                (6, int(blob.centroid[1]), int(blob.centroid[0]))
                for blob in connected_components(
                    state.frame(), colors=(9, 12, 13, 14, 15), min_area=2
                )
            ]
            for action in options:
                child = state.clone()
                child.step(*action)
                candidate_path = path + [action]
                horizontal = sum(value[0] for value in gaps(child).values())
                if child.levels_completed > base_level or horizontal < baseline:
                    stage = candidate_path
                    queue.clear()
                    break
                key = arr(child.frame())[1:].tobytes()
                if key not in seen:
                    seen.add(key)
                    queue.append((child, candidate_path))
        if stage is None:
            print("horizontal_stall", baseline, len(seen))
            break
        for action in stage:
            node.step(*action)
        full_path.extend(stage)
        trace.append(gaps(node))
        if node.levels_completed > base_level:
            break
    print("horizontal", node.levels_completed, len(full_path), full_path, trace)

    align_marker_pair_with_pressure_controls(
        node, marker_color=15, max_stages=24, max_states=1200, max_depth=18
    )
    print("align15", node.levels_completed, gaps(node))

    baseline = sum(value[0] for value in gaps(node).values())
    queue = deque([(node.clone(), [])])
    seen = {arr(node.frame())[1:].tobytes()}
    next_stage = None
    while queue and len(seen) <= 4000:
        state, path = queue.popleft()
        if len(path) >= 32:
            continue
        options = [
            (6, int(blob.centroid[1]), int(blob.centroid[0]))
            for blob in connected_components(
                state.frame(), colors=(9, 12, 13, 14, 15), min_area=2
            )
        ]
        for action in options:
            child = state.clone()
            child.step(*action)
            candidate_path = path + [action]
            horizontal = sum(value[0] for value in gaps(child).values())
            if child.levels_completed > base_level or horizontal < baseline:
                next_stage = (child.levels_completed, gaps(child), candidate_path)
                queue.clear()
                break
            key = arr(child.frame())[1:].tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((child, candidate_path))
    print("next_horizontal", len(seen), next_stage)


arena.run_program("vc33", probe)
