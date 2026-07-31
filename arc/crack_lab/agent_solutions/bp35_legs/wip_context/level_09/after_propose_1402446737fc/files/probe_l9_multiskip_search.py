"""Bounded best-first search from the viable three-skip frontier."""

import heapq
import itertools
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_multi_skip_bottom import root_for
from probe_l9_route_deletions import enter_level_9


MAX_STATES = 900
MAX_DEPTH = 23
MAX_SECONDS = 35.0


def avatar(env):
    blobs = [
        blob
        for blob in connected_components(
            env.frame(), colors=(9,), min_area=3
        )
        if blob.bbox[0] < 63
    ]
    if not blobs:
        return None
    return (
        round((blobs[0].centroid[0] - 3) / 6),
        round((blobs[0].centroid[1] - 3) / 6),
    )


def actions(env):
    result = [(3,), (4,), *controls(env)]
    cell = avatar(env)
    if cell is None:
        return tuple(result)
    ai, aj = cell
    for blob in connected_components(
        env.frame(), colors=(12, 14, 15), min_area=3
    ):
        if blob.bbox[0] >= 63 or blob.area != 21:
            continue
        bi = round((blob.centroid[0] - 3) / 6)
        bj = round((blob.centroid[1] - 3) / 6)
        distance = abs(bi - ai) + abs(bj - aj)
        if (blob.color == 14 and distance <= 5) or distance <= 2:
            result.append(
                (6, round(blob.centroid[1]), round(blob.centroid[0]))
            )
    return tuple(dict.fromkeys(result))


def physical_key(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return frame.tobytes()


def goal_score(env):
    frame = env.frame()
    cell = avatar(env)
    goals = [
        blob
        for blob in connected_components(frame, colors=(7,), min_area=2)
        if blob.bbox[0] < 63
    ]
    if cell is None or not goals:
        return 0, -99
    ai, aj = cell
    target = (
        round((goals[0].centroid[0] - 3) / 6),
        round((goals[0].centroid[1] - 3) / 6),
    )
    queue = [cell]
    seen = {cell}
    while queue:
        i, j = queue.pop()
        if (i, j) == target:
            return 2, 0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if not (0 <= ni < 10 and 0 <= nj < 10):
                continue
            if (ni, nj) in seen:
                continue
            color = int(frame[3 + 6 * ni][3 + 6 * nj])
            if color in (3, 5):
                continue
            seen.add((ni, nj))
            queue.append((ni, nj))
    return 1, -(abs(ai - target[0]) + abs(aj - target[1]))


def novelty_score(env):
    visible, distance = goal_score(env)
    cell = avatar(env)
    outside = int(cell is not None and cell[1] == 0)
    return visible, outside, distance, -len(controls(env))


def search(root):
    base_level = int(root.levels_completed)
    counter = itertools.count()
    initial_score = novelty_score(root)
    queue = [
        (
            tuple(-value for value in initial_score),
            0,
            next(counter),
            root.clone(),
            (),
        )
    ]
    seen = {physical_key(root)}
    started = time.monotonic()
    expanded = 0
    best = (initial_score, (), compact(root))
    while (
        queue
        and expanded < MAX_STATES
        and time.monotonic() - started < MAX_SECONDS
    ):
        _, _, _, node, path = heapq.heappop(queue)
        expanded += 1
        if expanded % 100 == 0:
            print(
                "PROGRESS",
                expanded,
                len(seen),
                len(queue),
                len(path),
                best,
                flush=True,
            )
        if len(path) >= MAX_DEPTH:
            continue
        for action in actions(node):
            child = node.clone()
            child.step(*action)
            child_path = path + (action,)
            if int(child.levels_completed) > base_level:
                return child_path, expanded, len(seen), best
            if child.terminal() or avatar(child) is None:
                continue
            key = physical_key(child)
            if key in seen:
                continue
            seen.add(key)
            score = novelty_score(child)
            if score > best[0]:
                best = (score, child_path, compact(child))
            heapq.heappush(
                queue,
                (
                    tuple(-value for value in score),
                    len(child_path),
                    next(counter),
                    child,
                    child_path,
                ),
            )
    return (), expanded, len(seen), best


def probe(env):
    enter_level_9(env)
    root = root_for(env, 3)
    path, expanded, seen, best = search(root)
    print(
        "SEARCH",
        "path",
        path,
        "expanded",
        expanded,
        "seen",
        seen,
        "best",
        best,
        flush=True,
    )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
