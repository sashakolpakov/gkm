"""Offset-independent best-first search from the verified level-7 entry."""

import heapq
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import run_actions
from perception import connected_components


LEFT = (3,)
RIGHT = (4,)
DOWN_CLICK = 6
ALT_CLICK = 7

SEED = [
    RIGHT,
    RIGHT,
    RIGHT,
    (6, 39, 51),
    (6, 3, 3),
    RIGHT,
    (6, 3, 3),
    RIGHT,
    LEFT,
    (7, 3, 3),
    (7, 3, 3),
    LEFT,
    (6, 3, 9),
    (6, 39, 35),
    (7, 3, 5),
    (6, 39, 35),
    (6, 39, 17),
    (6, 3, 5),
    (6, 39, 33),
    (6, 39, 51),
    (6, 3, 15),
    (6, 39, 5),
    LEFT,
    (6, 3, 11),
    (6, 3, 39),
    (6, 3, 23),
    (6, 3, 51),
    (6, 3, 35),
    RIGHT,
    (6, 3, 62),
    (6, 39, 35),
    (6, 3, 47),
    (7, 45, 38),
    (6, 3, 47),
    (7, 45, 38),
    (6, 39, 17),
    (6, 3, 47),
    (6, 39, 33),
]


def avatar_position(frame):
    pixels = np.asarray(frame)
    ys, xs = np.where(pixels == 9)
    if len(xs) == 0:
        return None
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def control_target(frame):
    controls = [
        blob
        for blob in connected_components(frame, colors=(8,), min_area=3)
        if blob.bbox[1] <= 5 and blob.bbox[0] < 63
    ]
    if not controls:
        return None
    blob = controls[0]
    return 3, int(round(blob.centroid[0]))


def support_targets(frame, ax, ay):
    pixels = np.asarray(frame)
    targets = set()
    for y in range(1, 62):
        for x in range(1, 63):
            if (
                int(pixels[y, x]) == 12
                and abs(x - ax) <= 3
                and abs(y - ay) <= 25
                and all(
                    int(pixels[y + dy, x + dx]) == 12
                    for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1))
                )
                and all(
                    int(pixels[y + dy, x + dx]) != 12
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
            ):
                targets.add((x, y))
    for blob in connected_components(frame, colors=(12, 14), min_area=6):
        x = int(round(blob.centroid[1]))
        y = int(round(blob.centroid[0]))
        if abs(x - ax) <= 3 and abs(y - ay) <= 25:
            targets.add((x, y))
    return sorted(targets, key=lambda target: abs(target[1] - ay))


def choices(node):
    frame = node.frame()
    avatar = avatar_position(frame)
    out = [LEFT, RIGHT, (ALT_CLICK, 0, 0)]
    if avatar is None:
        return out
    ax, ay = avatar
    control = control_target(frame)
    if control is not None:
        out.append((DOWN_CLICK, *control))
    for x, y in support_targets(frame, ax, ay):
        out.append((DOWN_CLICK, x, y))
    return list(dict.fromkeys(out))


def state_key(node, phase):
    return np.asarray(node.frame())[:63].tobytes(), phase


def target_distance(frame):
    pixels = np.asarray(frame)
    avatar = avatar_position(pixels)
    ys, xs = np.where(pixels[:63] == 7)
    if avatar is None or len(xs) == 0:
        return None
    ax, ay = avatar
    tx = int(round(float(xs.mean())))
    ty = int(round(float(ys.mean())))
    return (abs(ax - tx) + abs(ay - ty) + 5) // 6


def target_path_distance(frame):
    pixels = np.asarray(frame)
    avatar = avatar_position(pixels)
    ys, xs = np.where(pixels[:63] == 7)
    if avatar is None or len(xs) == 0:
        return None
    ax, ay = avatar
    row_centers = sorted(
        ay + 6 * offset
        for offset in range(-10, 11)
        if 2 <= ay + 6 * offset <= 60
    )
    col_centers = [15 + 6 * column for column in range(8)]
    start = (
        row_centers.index(ay),
        min(range(8), key=lambda column: abs(col_centers[column] - ax)),
    )
    tx = int(round(float(xs.mean())))
    ty = int(round(float(ys.mean())))
    target = (
        min(range(len(row_centers)), key=lambda row: abs(row_centers[row] - ty)),
        min(range(8), key=lambda column: abs(col_centers[column] - tx)),
    )

    def blocked(row, column):
        if (row, column) == target:
            return False
        y, x = row_centers[row], col_centers[column]
        patch = pixels[max(0, y - 3):min(63, y + 4), x - 3:x + 4]
        return (
            int(np.count_nonzero((patch == 3) | (patch == 5))) >= 10
            or int(np.count_nonzero(patch == 12)) >= 10
            or int(np.count_nonzero(patch == 15)) >= 3
        )

    queue = [(start, 0)]
    seen = {start}
    for (row, column), distance in queue:
        if (row, column) == target:
            return distance
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cell = row + dr, column + dc
            if (
                0 <= cell[0] < len(row_centers)
                and 0 <= cell[1] < 8
                and cell not in seen
                and not blocked(*cell)
            ):
                seen.add(cell)
                queue.append((cell, distance + 1))
    return len(row_centers) + 8


def search(env, max_states=1800, max_depth=120):
    run_actions(env, SEED)
    base_level = int(env.levels_completed)
    root = env.clone()
    counter = itertools.count()
    root_distance = target_path_distance(root.frame())
    root_heuristic = 8 if root_distance is None else root_distance
    frontier = [(0, 0, next(counter), root_distance, (), 0)]
    seen = {state_key(root, 0)}
    started = time.monotonic()
    best_distance = root_distance
    for expanded in range(1, max_states + 1):
        if not frontier:
            break
        _priority, _depth, _tie, distance, path, phase = heapq.heappop(frontier)
        node = root.clone()
        run_actions(node, path)
        if len(path) >= max_depth or node.terminal():
            continue
        before = np.asarray(node.frame()).copy()
        for action in choices(node):
            child = node.clone()
            child.step(*action)
            if child.levels_completed > base_level:
                route = list(SEED + list(path) + [action])
                print(
                    {
                        "found": True,
                        "expanded": expanded,
                        "seconds": round(time.monotonic() - started, 3),
                        "route_len": len(route),
                        "route": route,
                    },
                    flush=True,
                )
                return route
            if child.terminal() or avatar_position(child.frame()) is None:
                continue
            changed = not np.array_equal(before[:63], np.asarray(child.frame())[:63])
            control = action[0] == DOWN_CLICK and len(action) == 3 and action[1] <= 5
            if action[0] == ALT_CLICK:
                new_phase = 0
            else:
                new_phase = phase ^ bool(control and changed)
            child_frame = np.asarray(child.frame())
            new_distance = target_path_distance(child_frame)
            key = state_key(child, new_phase)
            if key in seen:
                continue
            seen.add(key)
            new_path = path + (action,)
            heapq.heappush(
                frontier,
                (
                    len(new_path),
                    len(new_path),
                    next(counter),
                    new_distance,
                    new_path,
                    new_phase,
                ),
            )
            if new_distance is not None and (
                best_distance is None or new_distance < best_distance
            ):
                best_distance = new_distance
                best_route = list(SEED) + list(new_path)
                print(
                    {
                        "target_distance": best_distance,
                        "expanded": expanded,
                        "depth": len(new_path),
                        "frontier": len(frontier),
                        "seconds": round(time.monotonic() - started, 3),
                        "route": best_route,
                    },
                    flush=True,
                )
        if expanded % 100 == 0:
            print(
                {
                    "expanded": expanded,
                    "best_distance": best_distance,
                    "frontier": len(frontier),
                    "seconds": round(time.monotonic() - started, 3),
                },
                flush=True,
            )
    print(
        {
            "found": False,
            "expanded": expanded,
            "best_distance": best_distance,
            "frontier": len(frontier),
            "seconds": round(time.monotonic() - started, 3),
        },
        flush=True,
    )
    return []


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    search(env)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
