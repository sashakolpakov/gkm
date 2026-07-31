import itertools
import json
import time

import gkm_try as H

from perception import connected_components
from probe_targets8 import center, groups


PREFIX = (
    (6, 7, 48), (6, 7, 54), (6, 15, 19), (6, 7, 19),
    (6, 16, 53), (6, 31, 44), (6, 42, 43),
    (6, 10, 53), (6, 7, 55),
)
BEAM = 120
MAX_DEPTH = 18


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start_level = int(env.levels_completed)
    ring_centers = tuple(
        (round(blob.centroid[0]), round(blob.centroid[1]))
        for blob in connected_components(
            env.frame(), colors=(9,), min_area=9
        )
        if blob.bbox[0] >= 10
    )
    root = env.clone()
    for action in PREFIX:
        root.step(*action)

    def features(node):
        frame = node.frame()
        squares = tuple(
            (
                blob.bbox,
                (
                    (blob.bbox[0] + blob.bbox[2]) // 2,
                    (blob.bbox[1] + blob.bbox[3]) // 2,
                ),
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
        stars = groups(frame, 14)
        cutters = groups(frame, 7)
        return squares, stars, cutters

    def valid(value):
        squares, stars, cutters = value
        return (
            len(squares) == 2
            and len(stars) == 1 and len(stars[0]) == 8
            and len(cutters) == 1 and len(cutters[0]) == 8
        )

    def target_centers(value):
        squares, stars, _ = value
        return (
            *(item_center for _, item_center in squares),
            center(stars[0]),
        )

    def assignment_distance(value):
        targets = target_centers(value)
        return min(
            sum(
                max(abs(row - target_row), abs(col - target_col))
                for (row, col), (target_row, target_col)
                in zip(targets, assignment)
            )
            for assignment in itertools.permutations(ring_centers, 3)
        )

    def score(value, depth):
        squares, stars, cutters = value
        star_center = center(stars[0])
        cutter_center = center(cutters[0])
        square_centers = tuple(item_center for _, item_center in squares)
        clearance = min(
            max(abs(mr - sr), abs(mc - sc))
            for mr, mc in (star_center, cutter_center)
            for sr, sc in square_centers
        )
        collision_penalty = max(0, 10 - clearance) * 4
        return assignment_distance(value) + collision_penalty + depth * 0.05

    def actions(value):
        squares, stars, cutters = value
        proposed = {(6, col, row) for row, col in ring_centers}
        for _, (row, col) in squares:
            proposed.add((6, col, row))
            for target_row, target_col in ring_centers:
                proposed.add((
                    6,
                    col + max(-6, min(6, target_col - col)),
                    row + max(-6, min(6, target_row - row)),
                ))
            for dr, dc in (
                (-6, 0), (6, 0), (0, -6), (0, 6),
                (-6, -6), (-6, 6), (6, -6), (6, 6),
            ):
                proposed.add((
                    6,
                    max(0, min(63, col + dc)),
                    max(10, min(62, row + dr)),
                ))
        proposed.update(
            (6, col, row)
            for group in stars + cutters
            for row, col in group
        )
        return tuple(sorted(proposed))

    root_features = features(root)
    print(
        "ROOT", target_centers(root_features),
        "cutter", center(root_features[2][0]),
        "distance", assignment_distance(root_features),
        flush=True,
    )
    beam = [(root, [], root_features, root_features)]
    started = time.monotonic()
    clone_steps = 0
    for depth in range(1, MAX_DEPTH + 1):
        candidates = {}
        for node, path, value, prior_value in beam:
            for action in actions(value):
                child = node.clone()
                child.step(*action)
                clone_steps += 1
                delay = clone_steps / 300 - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
                child_path = path + [action]
                if int(child.levels_completed) > start_level:
                    print("FOUND", PREFIX + tuple(child_path), flush=True)
                    return
                if child.terminal():
                    continue
                child_value = features(child)
                if not valid(child_value):
                    continue
                key = (
                    child_value[0],
                    child_value[1],
                    child_value[2],
                    prior_value[1],
                    prior_value[2],
                    action,
                )
                item = (child, child_path, child_value, value)
                previous = candidates.get(key)
                if (
                    previous is None
                    or score(child_value, depth)
                    < score(previous[2], depth)
                ):
                    candidates[key] = item
        ranked = sorted(
            candidates.values(),
            key=lambda item: score(item[2], depth),
        )
        beam = ranked[:BEAM]
        if not beam:
            break
        best = beam[0]
        print(
            "DEPTH", depth, "score", score(best[2], depth),
            "targets", target_centers(best[2]),
            "cutter", center(best[2][2][0]),
            "suffix", best[1], "states", len(candidates),
            "steps", clone_steps, flush=True,
        )
    print("NO_PATH", clone_steps, flush=True)


if __name__ == "__main__":
    H.A.run_program("su15", inspect)
