import itertools
import json
import time

import gkm_try as H

from perception import connected_components
from probe_clean8 import PREFIX, body_groups, body_pixels


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start_level = int(env.levels_completed)
    initial = env.frame()
    ring_masks = tuple(
        {
            (row, col)
            for row in range(blob.bbox[0], blob.bbox[2] + 1)
            for col in range(blob.bbox[1], blob.bbox[3] + 1)
            if int(initial[row][col]) == 9
        }
        for blob in connected_components(
            initial, colors=(9,), min_area=9
        )
        if blob.bbox[0] >= 10
    )
    targets = tuple(
        (
            round(sum(row for row, _ in mask) / len(mask)),
            round(sum(col for _, col in mask) / len(mask)),
        )
        for mask in ring_masks
    )
    root = env.clone()
    for action in PREFIX:
        root.step(*action)

    def key(node):
        square = tuple(
            blob.bbox
            for blob in connected_components(
                node.frame(), colors=(8,), min_area=9
            )
            if blob.bbox[0] >= 10
        )
        return square, body_groups(node.frame())

    def centers(groups):
        return tuple(
            (
                round(sum(row for row, _ in group) / len(group)),
                round(sum(col for _, col in group) / len(group)),
            )
            for group in groups
        )

    def score(node):
        groups = body_groups(node.frame())
        pixels = {point for group in groups for point in group}
        overlap = sum(point in mask for point in pixels for mask in ring_masks)
        agents = centers(groups)
        distance = min(
            sum(
                max(abs(row - tr), abs(col - tc))
                for (row, col), (tr, tc) in zip(agents, order)
            )
            for order in itertools.permutations(targets, len(agents))
        )
        return overlap, -distance

    beam = [(root, [])]
    seen = {key(root)}
    best = score(root)[0]
    clone_steps = 0
    started = time.monotonic()
    print("SEARCH_ROOT", score(root))
    for depth in range(1, 16):
        candidates = []
        for node, path in beam:
            actions = [
                (6, col, row)
                for row, col in sorted(body_pixels(node.frame()))
            ]
            actions.append((6, 32, 32))
            for action in actions:
                child = node.clone()
                child.step(*action)
                clone_steps += 1
                target_elapsed = clone_steps / 300
                remaining = target_elapsed - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
                child_path = path + [action]
                if int(child.levels_completed) > start_level:
                    print("FOUND", PREFIX + child_path)
                    return
                if child.terminal():
                    continue
                child_key = key(child)
                if child_key in seen:
                    continue
                seen.add(child_key)
                candidates.append((score(child), child, child_path))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = []
        bucket_counts = {}
        for item in candidates:
            overlap = item[0][0]
            if bucket_counts.get(overlap, 0) >= 4:
                continue
            bucket_counts[overlap] = bucket_counts.get(overlap, 0) + 1
            selected.append(item)
            if len(selected) >= 40:
                break
        beam = [(node, path) for _, node, path in selected]
        if not beam:
            break
        if candidates[0][0][0] > best:
            best = candidates[0][0][0]
            print(
                "NEW_BEST",
                depth,
                candidates[0][0],
                "suffix",
                candidates[0][2],
                flush=True,
            )
        print(
            "DEPTH",
            depth,
            "best",
            candidates[0][0],
            "beam",
            len(beam),
            "seen",
            len(seen),
            flush=True,
        )
    print("NO_PATH", len(seen), best)


H.A.run_program("su15", inspect)
