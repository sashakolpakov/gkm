"""Forward state search from the verified top room, using undo only to backtrack."""

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import (
    COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action, moves_used,
    run_actions,
)
from perception import connected_components
from probe_l7_fresh_graph import signed_origin_delta
from probe_l7_raw_search import target_path_distance
from probe_level7_reward_recovery import PREFIX, SUFFIX, avatar_cell


LEFT, RIGHT, UNDO = (3,), (4,), (7,)
TOP_ROUTE = [
    *PREFIX,
    click_action(5, 2),
    *SUFFIX,
    LEFT, (6, 3, 9), RIGHT, (6, 3, 39),
    LEFT, LEFT, LEFT,
]


def digest(frame):
    return hashlib.blake2b(
        np.asarray(frame)[:63].tobytes(), digest_size=12
    ).digest()


def target_cell(frame):
    ys, xs = np.where(np.asarray(frame)[:63] == 7)
    if not len(xs):
        return None
    y, x = float(ys.mean()), float(xs.mean())
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - y)),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - x)),
    )


def gravity_actions(frame):
    actions = []
    for blob in connected_components(frame, colors=(8,), min_area=1):
        if blob.bbox[0] >= 63 or blob.bbox[1] > 5:
            continue
        y, x = blob.centroid
        action = (6, round(x), round(y))
        if action not in actions:
            actions.append(action)
    return actions


def choices(frame):
    out = [LEFT, RIGHT, *gravity_actions(frame)]
    for i in range(10):
        for j in range(8):
            color, _area = _cell_shape(frame, i, j)
            if color in (12, 14):
                out.append(click_action(i, j))
    return list(dict.fromkeys(out))


def expanded_supports(frame):
    return sum(
        _cell_shape(frame, i, j)[0] == 12
        and _cell_shape(frame, i, j)[1] >= 13
        for i in range(10)
        for j in range(8)
    )


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    run_actions(env, TOP_ROUTE)
    if env.terminal():
        print("TOP_INPLACE_ROOT_DEAD", flush=True)
        return

    base_level = int(env.levels_completed)
    root = env.clone()
    node = root.clone()
    max_states = int(os.environ.get("MAX_STATES", "600"))
    max_depth = int(os.environ.get("MAX_DEPTH", "18"))
    expanded = 0
    steps = 0
    best_depth = {}
    best = None
    best_path = ()

    def key(frame, origin):
        return digest(frame), moves_used(frame) % 2, origin

    def metric(frame, origin, depth):
        target = target_cell(frame)
        distance = target_path_distance(frame)
        avatar = avatar_cell(frame)
        return (
            0 if target is not None else 1,
            99 if distance is None else distance,
            -abs(origin),
            99 if avatar is None else -avatar[1],
            -expanded_supports(frame),
            depth,
        )

    def reconstruct(path):
        fresh = root.clone()
        run_actions(fresh, path)
        return fresh

    root_frame = np.asarray(node.frame()).copy()
    best_depth[key(root_frame, 0)] = 0
    best = metric(root_frame, 0, 0)
    print(
        "TOP_INPLACE_ROOT", avatar_cell(root_frame), target_cell(root_frame),
        tuple(gravity_actions(root_frame)), best, flush=True,
    )

    def dfs(path, origin):
        nonlocal node, expanded, steps, best, best_path
        if expanded >= max_states or len(path) >= max_depth:
            return None
        before = np.asarray(node.frame()).copy()
        before_digest = digest(before)
        ranked = []
        effect_keys = set()
        for action in choices(before):
            node.step(*action)
            steps += 1
            if node.levels_completed > base_level:
                return (*path, action)
            if not node.terminal() and avatar_cell(node.frame()) is not None:
                child_frame = np.asarray(node.frame()).copy()
                delta = signed_origin_delta(before, child_frame)
                child_origin = origin + delta
                child_key = key(child_frame, child_origin)
                child_metric = metric(child_frame, child_origin, len(path) + 1)
                effect = (child_key, action[0] in (3, 4))
                if effect not in effect_keys:
                    effect_keys.add(effect)
                    ranked.append(
                        (child_metric, -abs(delta), action,
                         child_origin, child_key)
                    )
                node.step(*UNDO)
                steps += 1
                if node.terminal() or digest(node.frame()) != before_digest:
                    node = reconstruct(path)
            else:
                node = reconstruct(path)

        ranked.sort()
        for child_metric, _shift, action, child_origin, child_key in ranked:
            prior = best_depth.get(child_key)
            if prior is not None and prior <= len(path) + 1:
                continue
            node.step(*action)
            steps += 1
            if node.levels_completed > base_level:
                return (*path, action)
            if node.terminal() or avatar_cell(node.frame()) is None:
                node = reconstruct(path)
                continue
            expanded += 1
            child_path = (*path, action)
            best_depth[child_key] = len(child_path)
            if child_metric < best:
                best = child_metric
                best_path = child_path
                print(
                    "TOP_INPLACE_PROGRESS", expanded, steps, best,
                    avatar_cell(node.frame()), target_cell(node.frame()),
                    child_path, flush=True,
                )
            if expanded % 100 == 0:
                print(
                    "TOP_INPLACE_SEARCH", expanded, steps, len(best_depth),
                    best, flush=True,
                )
            winner = dfs(child_path, child_origin)
            if winner is not None:
                return winner
            node.step(*UNDO)
            steps += 1
            if node.terminal() or digest(node.frame()) != before_digest:
                node = reconstruct(path)
            if expanded >= max_states:
                break
        return None

    winner = dfs((), 0)
    if winner is None:
        print(
            "TOP_INPLACE_DONE", expanded, steps, len(best_depth),
            best, best_path, flush=True,
        )
        return
    verified = root.clone()
    run_actions(verified, winner)
    print(
        "TOP_INPLACE_WIN", winner, int(verified.levels_completed),
        bool(verified.terminal()), expanded, steps, flush=True,
    )
    if verified.levels_completed > base_level:
        print("TOP_INPLACE_ROUTE", [*TOP_ROUTE, *winner], flush=True)


arena.run_program("bp35", probe)
