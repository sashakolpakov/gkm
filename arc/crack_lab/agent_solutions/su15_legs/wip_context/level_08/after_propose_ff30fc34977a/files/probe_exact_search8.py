import itertools
import json
import time

import gkm_try as H

from perception import connected_components
from probe_clean8 import PREFIX, body_groups, body_pixels


BEAM = 400
MAX_DEPTH = 14


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start_level = int(env.levels_completed)
    initial = env.frame()
    ring_masks = tuple(
        frozenset(
            (row, col)
            for row in range(blob.bbox[0], blob.bbox[2] + 1)
            for col in range(blob.bbox[1], blob.bbox[3] + 1)
            if int(initial[row][col]) == 9
        )
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )[1:]

    root = env.clone()
    for action in PREFIX:
        root.step(*action)

    def group_distance(group, mask):
        return min(
            (
                max(abs(dr), abs(dc))
                for dr in range(-12, 13)
                for dc in range(-12, 13)
                if all((row + dr, col + dc) in mask for row, col in group)
            ),
            default=99,
        )

    def metric(groups):
        distances = tuple(
            tuple(group_distance(group, mask) for mask in ring_masks)
            for group in groups
        )
        exact_distance = min(
            sum(distances[index][target] for index, target in enumerate(order))
            for order in itertools.permutations(range(3))
        )
        pixels = {point for group in groups for point in group}
        overlap = sum(point in mask for point in pixels for mask in ring_masks)
        return exact_distance, -overlap

    root_groups = body_groups(root.frame())
    print("ROOT", metric(root_groups), flush=True)
    beam = [(root, [], root_groups)]
    clone_steps = 0
    started = time.monotonic()
    for depth in range(1, MAX_DEPTH + 1):
        candidates = {}
        for node, path, prior_groups in beam:
            actions = [
                (6, col, row)
                for row, col in sorted(body_pixels(node.frame()))
            ]
            for action in actions:
                child = node.clone()
                child.step(*action)
                clone_steps += 1
                remaining = clone_steps / 300 - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
                child_path = path + [action]
                if int(child.levels_completed) > start_level:
                    print("FOUND", PREFIX + child_path, flush=True)
                    return
                if child.terminal():
                    continue
                groups = body_groups(child.frame())
                if len(groups) != 3:
                    continue
                key = (groups, prior_groups)
                item = (metric(groups), child, child_path, groups)
                previous = candidates.get(key)
                if previous is None or item[0] < previous[0]:
                    candidates[key] = item
        ranked = sorted(candidates.values(), key=lambda item: item[0])
        selected = []
        distance_counts = {}
        for item in ranked:
            distance = item[0][0]
            if distance_counts.get(distance, 0) >= 100:
                continue
            distance_counts[distance] = distance_counts.get(distance, 0) + 1
            selected.append(item)
            if len(selected) >= BEAM:
                break
        beam = [
            (node, path, groups)
            for _, node, path, groups in selected
        ]
        if not beam:
            break
        print(
            "DEPTH",
            depth,
            "best",
            selected[0][0],
            "beam",
            len(beam),
            "states",
            len(candidates),
            "suffix",
            selected[0][2],
            flush=True,
        )
    print("NO_PATH", clone_steps, flush=True)


H.A.run_program("su15", inspect)
