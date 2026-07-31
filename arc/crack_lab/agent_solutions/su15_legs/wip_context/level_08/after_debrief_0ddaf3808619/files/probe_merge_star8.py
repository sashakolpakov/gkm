import heapq
import itertools
import json
import time

import gkm_try as H

from perception import connected_components
from probe_targets8 import center, groups


STAGE = (
    (6, 7, 48),
    (6, 7, 54),
    (6, 15, 19),
    (6, 7, 19),
)
BEAM = 140
MAX_DEPTH = 3


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
    right_rings = tuple(point for point in ring_centers if point[1] > 32)
    root = env.clone()
    for action in STAGE:
        root.step(*action)

    def solids(node):
        return tuple(
            (blob.color, blob.bbox)
            for blob in connected_components(node.frame(), min_area=1)
            if (
                blob.bbox[0] >= 10
                and blob.size[0] == blob.size[1]
                and blob.area == blob.size[0] ** 2
                and blob.color not in (3, 4, 5, 7, 9)
            )
        )

    def valid(node):
        items = solids(node)
        stars = groups(node.frame(), 14)
        return (
            sum(color == 8 for color, _ in items) == 1
            and sum(color == 12 for color, _ in items) == 1
            and all(len(star) == 8 for star in stars)
        )

    def state(node):
        frame = node.frame()
        return (
            solids(node),
            tuple((7, center(group)) for group in groups(frame, 7)),
            tuple((14, center(group)) for group in groups(frame, 14)),
        )

    def priority(node):
        frame = node.frame()
        sevens = groups(frame, 7)
        stars = groups(frame, 14)
        if (
            len(stars) == 1 and len(stars[0]) == 8
            and len(sevens) == 1
        ):
            row, col = center(stars[0])
            current_distance = min(
                max(abs(row - tr), abs(col - tc))
                for tr, tc in right_rings
            )
            forecast = node.clone()
            forecast.step(6, 32, 10)
            if not valid(forecast):
                return 997, 997
            next_stars = groups(forecast.frame(), 14)
            if len(next_stars) != 1 or len(next_stars[0]) != 8:
                return 998, 998
            next_row, next_col = center(next_stars[0])
            next_distance = min(
                max(abs(next_row - tr), abs(next_col - tc))
                for tr, tc in right_rings
            )
            heading_penalty = 20 if next_distance >= current_distance else 0
            return next_distance + heading_penalty, -next_col
        if len(sevens) != 3 or stars:
            return 999, 999
        centers = tuple(center(group) for group in sevens)
        pair_options = []
        for left, right in itertools.combinations(centers, 2):
            distance = max(
                abs(left[0] - right[0]), abs(left[1] - right[1])
            )
            midpoint = (
                round((left[0] + right[0]) / 2),
                round((left[1] + right[1]) / 2),
            )
            ring_distance = min(
                max(abs(midpoint[0] - tr), abs(midpoint[1] - tc))
                for tr, tc in right_rings
            )
            pair_options.append(distance + ring_distance * 0.15)
        return 100 + min(pair_options), 0

    def actions(node):
        frame = node.frame()
        proposed = {
            (6, 0, 0),
            *((6, col, row) for row, col in ring_centers),
        }
        for color in (7, 14):
            proposed.update(
                (6, col, row)
                for group in groups(frame, color)
                for row, col in group
            )
        for blob in connected_components(
            frame, colors=(8, 12), min_area=25
        ):
            if blob.bbox[0] < 10:
                continue
            row, col = center(tuple(
                (r, c)
                for r in range(blob.bbox[0], blob.bbox[2] + 1)
                for c in range(blob.bbox[1], blob.bbox[3] + 1)
            ))
            for dr, dc in (
                (0, 0), (-6, 0), (6, 0), (0, -6), (0, 6),
                (-6, -6), (-6, 6), (6, -6), (6, 6),
            ):
                proposed.add((
                    6,
                    max(0, min(63, col + dc)),
                    max(10, min(62, row + dr)),
                ))
        return tuple(sorted(proposed))

    serial = itertools.count()
    beam = [(priority(root), next(serial), root, [], state(root))]
    started = time.monotonic()
    clone_steps = 0
    for depth in range(1, MAX_DEPTH + 1):
        candidates = {}
        for _, _, node, path, prior_state in beam:
            for action in actions(node):
                child = node.clone()
                child.step(*action)
                clone_steps += 1
                delay = clone_steps / 300 - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
                child_path = path + [action]
                if int(child.levels_completed) > start_level:
                    print("FOUND", STAGE + tuple(child_path), flush=True)
                    return
                if child.terminal() or not valid(child):
                    continue
                child_state = state(child)
                key = (child_state, prior_state, action)
                item = (
                    priority(child), next(serial), child,
                    child_path, child_state,
                )
                previous = candidates.get(key)
                if previous is None or item[0] < previous[0]:
                    candidates[key] = item
        ranked = sorted(candidates.values(), key=lambda item: item[0])
        beam = ranked[:BEAM]
        if not beam:
            break
        print(
            "DEPTH", depth, "best", beam[0][0],
            "state", beam[0][4], "suffix", beam[0][3],
            "states", len(candidates), "steps", clone_steps,
            flush=True,
        )
        if beam[0][0][0] <= 2:
            print("STAR_PATH", STAGE + tuple(beam[0][3]), flush=True)
            return
    print("NO_PATH", clone_steps, flush=True)


if __name__ == "__main__":
    H.A.run_program("su15", inspect)
