import time

import numpy as np

import gkm_try as H

from perception import connected_components


PREFIX = (
    (6, 7, 48),
    (6, 7, 54),
    (6, 15, 19),
    (6, 7, 19),
    (6, 16, 53),
    (6, 31, 44),
    (6, 42, 43),
    (6, 10, 53),
    (6, 7, 55),
    (6, 37, 49),
    (6, 8, 19),
    (6, 8, 31),
    (6, 10, 42),
    (6, 20, 43),
    (6, 21, 43),
    (6, 30, 50),
    (6, 39, 49),
    (6, 40, 51),
)
BEAM = 64
MAX_DEPTH = 18


def inspect(env):
    H.resumed_solve(env)
    start_level = int(env.levels_completed)
    initial = np.asarray(env.frame())
    rings = tuple(
        (
            (round(blob.centroid[0]), round(blob.centroid[1])),
            frozenset(
                (row, col)
                for row in range(blob.bbox[0], blob.bbox[2] + 1)
                for col in range(blob.bbox[1], blob.bbox[3] + 1)
                if int(initial[row, col]) == 9
            ),
        )
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )
    right_rings = tuple(item for item in rings if item[0][1] > 32)
    left_masks = tuple(mask for (row, col), mask in rings if col < 32)

    root = env.clone()
    for action in PREFIX:
        root.step(*action)

    def points(node, color):
        frame = np.asarray(node.frame())
        rows, cols = np.where(frame[10:] == color)
        return frozenset(
            (int(row + 10), int(col)) for row, col in zip(rows, cols)
        )

    def center(value):
        return (
            round(sum(row for row, _ in value) / len(value)),
            round(sum(col for _, col in value) / len(value)),
        )

    def value(node):
        frame = np.asarray(node.frame())
        star = points(node, 14)
        cutter = points(node, 7)
        square_points = frozenset(
            (int(row + 10), int(col))
            for row, col in zip(*np.where(frame[10:] == 12))
        )
        if len(star) != 8 or len(cutter) != 8 or len(square_points) != 50:
            return None
        square_components = tuple(
            frozenset(
                (row, col)
                for row in range(blob.bbox[0], blob.bbox[2] + 1)
                for col in range(blob.bbox[1], blob.bbox[3] + 1)
            )
            for blob in connected_components(
                frame, colors=(12,), min_area=25
            )
            if (
                blob.bbox[0] >= 10
                and blob.size == (5, 5)
                and blob.area == 25
            )
        )
        if (
            len(square_components) != 2
            or sorted(
                max(len(square & mask) for mask in left_masks)
                for square in square_components
            ) != [25, 25]
        ):
            return None
        return star, cutter, square_components

    def score(current):
        star, cutter, squares = current
        star_center = center(star)
        cutter_center = center(cutter)
        overlap = max(len(star & mask) for _, mask in right_rings)
        distance = min(
            max(abs(star_center[0] - row), abs(star_center[1] - col))
            for (row, col), _ in right_rings
        )
        clearance = min(
            max(abs(cutter_center[0] - other[0]),
                abs(cutter_center[1] - other[1]))
            for other in (star_center, *(center(square) for square in squares))
        )
        collision_penalty = max(0, 10 - clearance) * 2
        return (
            overlap,
            -(distance + collision_penalty),
            min(clearance, 16),
            star_center[1],
        )

    root_value = value(root)
    print(
        "ROOT", "star", center(root_value[0]),
        "cutter", center(root_value[1]), "score", score(root_value),
        flush=True,
    )
    beam = [(root, (), root_value, None)]
    clone_steps = 0
    started = time.monotonic()
    for depth in range(1, MAX_DEPTH + 1):
        candidates = {}
        for node, path, current, prior_star in beam:
            for row, col in current[0] | current[1]:
                if row >= 63 or col <= 0 or col >= 63:
                    continue
                action = (6, col, row)
                child = node.clone()
                child.step(*action)
                clone_steps += 1
                delay = clone_steps / 300 - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
                child_path = path + (action,)
                if int(child.levels_completed) > start_level:
                    print("FOUND", PREFIX + child_path, flush=True)
                    return
                if child.terminal():
                    continue
                child_value = value(child)
                if child_value is None:
                    continue
                if (
                    child_value[0] == current[0]
                    and child_value[1] == current[1]
                    and child_value[2] == current[2]
                ):
                    continue
                key = (
                    child_value[0],
                    child_value[1],
                    current[0],
                    current[1],
                )
                item = (child, child_path, child_value, current[0])
                old = candidates.get(key)
                if old is None or score(child_value) > score(old[2]):
                    candidates[key] = item
        ranked = sorted(
            candidates.values(), key=lambda item: score(item[2]), reverse=True
        )
        beam = []
        buckets = {}
        for item in ranked:
            star_center = center(item[2][0])
            prior_center = center(item[3])
            motion = (
                np.sign(star_center[0] - prior_center[0]),
                np.sign(star_center[1] - prior_center[1]),
            )
            bucket = (star_center, motion)
            if buckets.get(bucket, 0) >= 8:
                continue
            buckets[bucket] = buckets.get(bucket, 0) + 1
            beam.append(item)
            if len(beam) >= BEAM:
                break
        if not beam:
            break
        best = beam[0]
        print(
            "DEPTH", depth,
            "star", center(best[2][0]),
            "cutter", center(best[2][1]),
            "score", score(best[2]),
            "states", len(candidates),
            "steps", clone_steps,
            "suffix", best[1],
            flush=True,
        )
    print("NO_PATH", clone_steps, flush=True)


H.A.run_program("su15", inspect)
