import heapq
import importlib.util
import itertools
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    _body_center,
    _body_groups,
    _click,
    _move_square_one_step,
    _solid_playfield_squares,
)
from perception import connected_components


reason = G._workspace_taint_reason(os.getcwd())
if reason:
    raise SystemExit(f"TAINTED WORKSPACE: {reason}")


def load_level(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(*action)


def rings(env):
    return tuple(
        tuple(map(round, blob.centroid))
        for blob in connected_components(env.frame(), colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )


def bodies(env):
    return tuple(
        (
            color,
            tuple(
                (
                    _body_center(group),
                    tuple(
                        (row - _body_center(group)[0], col - _body_center(group)[1])
                        for row, col in group
                    ),
                )
                for group in _body_groups(env, color)
            ),
        )
        for color in (7, 14, 13)
    )


def solids(env):
    return tuple(
        (blob.color, blob.bbox)
        for blob in _solid_playfield_squares(
            env, colors=(6, 8, 10, 11, 12, 15)
        )
    )


def key(env):
    return bodies(env), solids(env)


def actions(env, targets):
    proposed = [(6, 32, 32)]
    for color in (7, 14, 13):
        for group in _body_groups(env, color):
            proposed.extend((6, col, row) for row, col in group)
    proposed.append((6, 11, 41))
    return tuple(dict.fromkeys(proposed))


def score(env, depth, targets):
    final = _body_groups(env, 13)
    if final:
        row, col = _body_center(final[0])
        return max(abs(row - 41), abs(col - 11)) + depth
    merged = _body_groups(env, 14)
    if len(merged) == 2:
        left = _body_center(merged[0])
        right = _body_center(merged[1])
        return 100 + max(abs(left[0] - right[0]), abs(left[1] - right[1])) + depth
    return 1000 + depth


def stage(env):
    target_small = (55, 11)
    target_large = (55, 53)
    _click(env, 46, 18)
    for color, target in ((15, target_small), (8, target_large)):
        for _ in range(3):
            candidates = _solid_playfield_squares(env, colors=(color,))
            if not candidates:
                break
            square = candidates[0]
            row, col = map(round, square.centroid)
            if max(abs(row - target[0]), abs(col - target[1])) <= 1:
                break
            _move_square_one_step(env, square, target)


def program(env):
    load_level(env)
    start_level = env.levels_completed
    targets = rings(env)
    env.step(6, 56, 33)
    stage(env)
    for action in (
        (6, 9, 38),
        (6, 50, 41),
        (6, 50, 29),
        (6, 50, 17),
    ):
        env.step(*action)
    print("STAGED", key(env), flush=True)
    serial = itertools.count()
    start = env.clone()
    heap = [(score(start, 0, targets), 0, next(serial), start, [])]
    seen = {key(start)}
    expanded = 0
    while heap and expanded < 4000:
        _, depth, _, node, path = heapq.heappop(heap)
        expanded += 1
        if node.levels_completed > start_level:
            print("WIN", path, key(node), "EXPANDED", expanded, flush=True)
            return
        if depth >= 24:
            continue
        for action in actions(node, targets):
            child = node.clone()
            child.step(*action)
            if child.terminal() and child.levels_completed == start_level:
                continue
            pieces = _solid_playfield_squares(child, colors=(8, 15))
            if (
                len([blob for blob in pieces if blob.color == 8]) != 1
                or len([blob for blob in pieces if blob.color == 15]) != 1
            ):
                continue
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            child_path = path + [action]
            heapq.heappush(
                heap,
                (
                    score(child, len(child_path), targets),
                    len(child_path),
                    next(serial),
                    child,
                    child_path,
                ),
            )
    print("DONE", "EXPANDED", expanded, "SEEN", len(seen), flush=True)


print("RUN", A.run_program("su15", program)[0])
