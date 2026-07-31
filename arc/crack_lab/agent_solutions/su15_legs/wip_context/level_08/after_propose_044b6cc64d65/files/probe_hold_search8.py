import itertools
import json
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components
from probe_clean8 import PREFIX, body_groups


BEAM = 140
MAX_DEPTH = 24


def center(group):
    return (
        round(sum(row for row, _ in group) / len(group)),
        round(sum(col for _, col in group) / len(group)),
    )


def controls(groups):
    return tuple(dict.fromkeys(
        (6, col, row)
        for group in groups
        for row, col in tuple(group) + (center(group),)
    ))


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start_level = int(env.levels_completed)
    initial = env.frame()
    masks = tuple(
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

    root_groups = body_groups(root.frame())
    template_center = center(root_groups[0])
    offsets = tuple(
        (row - template_center[0], col - template_center[1])
        for row, col in root_groups[0]
    )
    valid_centers = tuple(
        tuple(
            (row, col)
            for row in range(10, 64)
            for col in range(64)
            if all((row + dr, col + dc) in mask for dr, dc in offsets)
        )
        for mask in masks
    )

    def score(groups):
        assignments = []
        group_sets = tuple(set(group) for group in groups)
        group_centers = tuple(center(group) for group in groups)
        for order in itertools.permutations(range(3)):
            overlaps = tuple(
                len(group & masks[target])
                for group, target in zip(group_sets, order)
            )
            distances = tuple(
                min(
                    max(abs(row - target_row), abs(col - target_col))
                    for target_row, target_col in valid_centers[target]
                )
                for (row, col), target in zip(group_centers, order)
            )
            assignments.append((
                sum(value == 8 for value in overlaps),
                sum(overlaps),
                -sum(distances),
                -max(distances),
            ))
        return max(assignments)

    beam = [(root, (), root_groups, root_groups)]
    clone_steps = 0
    started = time.monotonic()
    print("ROOT", score(root_groups), tuple(map(center, root_groups)), flush=True)
    for depth in range(1, MAX_DEPTH + 1):
        unique = {}
        for node, path, groups, prior_groups in beam:
            for action in controls(groups):
                child = node.clone()
                child.step(*action)
                clone_steps += 1
                delay = clone_steps / 300 - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
                child_path = path + (action,)
                if int(child.levels_completed) > start_level:
                    print("FOUND", PREFIX + list(child_path), flush=True)
                    return
                if child.terminal():
                    continue
                child_groups = body_groups(child.frame())
                if len(child_groups) != 3:
                    continue
                selected = next(
                    (
                        (center(group), action[2] - center(group)[0],
                         action[1] - center(group)[1])
                        for group in groups
                        if action[2:] == ()
                        or (action[2], action[1]) in set(group) | {center(group)}
                    ),
                    (None, 0, 0),
                )
                key = (child_groups, groups, prior_groups, selected[1:])
                item = (score(child_groups), child, child_path,
                        child_groups, groups)
                old = unique.get(key)
                if old is None or item[0] > old[0]:
                    unique[key] = item
        ranked = sorted(unique.values(), key=lambda item: item[0], reverse=True)
        selected_items = []
        buckets = {}
        for item in ranked:
            bucket = item[0][:2]
            if buckets.get(bucket, 0) >= 24:
                continue
            buckets[bucket] = buckets.get(bucket, 0) + 1
            selected_items.append(item)
            if len(selected_items) >= BEAM:
                break
        beam = [
            (node, path, groups, prior_groups)
            for _, node, path, groups, prior_groups in selected_items
        ]
        if not beam:
            break
        best = selected_items[0]
        print(
            "DEPTH", depth, "BEST", best[0],
            tuple(map(center, best[3])),
            "BEAM", len(beam), "STATES", len(unique),
            "SUFFIX", list(best[2]), flush=True,
        )
    print("NO_PATH", clone_steps, flush=True)


A.run_program("su15", inspect)
