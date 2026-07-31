import itertools
import json
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components
from probe_clean8 import PREFIX, body_groups


OFFSETS = (
    (-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 0),
    (0, 2), (1, -1), (1, 0), (1, 1),
)
SEED = (
    (1, (1, 1)), (2, (1, 0)), (0, (0, 2)), (1, (1, 1)),
    (2, (0, -2)), (0, (0, 2)), (2, (1, 1)), (1, (1, 1)),
)


def center(group):
    return (
        round(sum(row for row, _ in group) / len(group)),
        round(sum(col for _, col in group) / len(group)),
    )


def labeled(groups):
    groups = list(groups)
    top = min(groups, key=lambda group: center(group)[0])
    groups.remove(top)
    left = min(groups, key=lambda group: center(group)[1])
    groups.remove(left)
    return top, groups[0], left


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start_level = int(env.levels_completed)
    initial = env.frame()
    masks = tuple(
        {
            (row, col)
            for row in range(blob.bbox[0], blob.bbox[2] + 1)
            for col in range(blob.bbox[1], blob.bbox[3] + 1)
            if int(initial[row][col]) == 9
        }
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )[1:]
    root = env.clone()
    for action in PREFIX:
        root.step(*action)

    def overlap(groups):
        if len(groups) != 3:
            return -1
        pixels = {point for group in groups for point in group}
        return sum(point in mask for point in pixels for mask in masks)

    controls = tuple(itertools.product(range(3), OFFSETS))
    variants = {SEED}
    for index in range(len(SEED)):
        variants.add(SEED[:index] + SEED[index + 1:])
        for control in controls:
            variants.add(SEED[:index] + (control,) + SEED[index + 1:])
    for index in range(len(SEED) + 1):
        for control in controls:
            variants.add(SEED[:index] + (control,) + SEED[index:])
    for index in range(len(SEED) - 1):
        swapped = list(SEED)
        swapped[index], swapped[index + 1] = (
            swapped[index + 1], swapped[index]
        )
        variants.add(tuple(swapped))

    clone_steps = 0
    started = time.monotonic()

    def pace():
        delay = clone_steps / 300 - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)

    endpoints = []
    for genome in variants:
        node = root.clone()
        actions = []
        valid = True
        for label, offset in genome:
            groups = body_groups(node.frame())
            if len(groups) != 3:
                valid = False
                break
            group = labeled(groups)[label]
            row, col = center(group)
            action = (6, col + offset[1], row + offset[0])
            node.step(*action)
            actions.append(action)
            clone_steps += 1
            pace()
            if int(node.levels_completed) > start_level:
                print("FOUND", PREFIX + actions, flush=True)
                return
        groups = body_groups(node.frame())
        if valid and overlap(groups) >= 23:
            endpoints.append((node, tuple(actions), groups, groups))
    print("ENDPOINTS", len(endpoints), "steps", clone_steps, flush=True)

    frontier = endpoints
    for depth in range(1, 7):
        candidates = []
        for node, actions, groups, prior_groups in frontier:
            for group in groups:
                row, col = center(group)
                for offset in OFFSETS:
                    action = (6, col + offset[1], row + offset[0])
                    child = node.clone()
                    child.step(*action)
                    clone_steps += 1
                    pace()
                    child_actions = actions + (action,)
                    if int(child.levels_completed) > start_level:
                        print(
                            "FOUND", PREFIX + list(child_actions),
                            "depth", depth, "steps", clone_steps, flush=True,
                        )
                        return
                    if child.terminal():
                        continue
                    child_groups = body_groups(child.frame())
                    if len(child_groups) != 3:
                        continue
                    candidates.append((
                        overlap(child_groups), child, child_actions,
                        child_groups, groups, prior_groups, offset,
                    ))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if depth < 2:
            selected = candidates
        else:
            selected = []
            seen = set()
            buckets = {}
            for item in candidates:
                key = (item[3], item[4], item[5], item[6])
                if key in seen:
                    continue
                if buckets.get(item[0], 0) >= 300:
                    continue
                seen.add(key)
                buckets[item[0]] = buckets.get(item[0], 0) + 1
                selected.append(item)
                if len(selected) >= 1200:
                    break
        frontier = [
            (node, actions, groups, prior_groups)
            for _, node, actions, groups, prior_groups, _, _ in selected
        ]
        print(
            "DEPTH", depth, "best",
            candidates[0][0] if candidates else None,
            "frontier", len(frontier), "candidates", len(candidates),
            "steps", clone_steps, flush=True,
        )
        if not frontier:
            break
    print("NO_PATH", clone_steps, flush=True)


A.run_program("su15", inspect)
