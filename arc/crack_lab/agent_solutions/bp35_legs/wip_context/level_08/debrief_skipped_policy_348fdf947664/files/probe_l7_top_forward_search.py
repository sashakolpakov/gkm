"""Bounded forward-only graph search from the verified level-7 top state."""

import hashlib
import heapq
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import (
    ROW_ANCHORS, _cell_shape, click_action, moves_used, run_actions,
)
from probe_l7_decode_matrix import controls, target
from probe_l7_fresh_graph import signed_origin_delta
from probe_l7_raw_search import target_path_distance
from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import avatar_cell


LEFT, RIGHT = (3,), (4,)


def frame_digest(frame):
    return hashlib.blake2b(
        np.asarray(frame)[:63].tobytes(), digest_size=12
    ).digest()


def action_choices(frame):
    avatar = avatar_cell(frame)
    if avatar is None:
        return []
    ai, _aj = avatar
    out = [LEFT, RIGHT, *controls(frame)]
    for i in range(max(0, ai - 5), min(10, ai + 6)):
        for j in range(8):
            if _cell_shape(frame, i, j)[0] in (12, 14):
                out.append(click_action(i, j))
    return list(dict.fromkeys(out))


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    run_actions(env, decoded_route())
    if env.terminal():
        print("TOP_FORWARD_ROOT_DEAD", flush=True)
        return
    base_level = int(env.levels_completed)
    root = env.clone()
    root_frame = np.asarray(root.frame()).copy()
    max_states = int(os.environ.get("MAX_STATES", "600"))
    max_depth = int(os.environ.get("MAX_DEPTH", "24"))
    counter = itertools.count()

    def key(node, origin):
        frame = node.frame()
        return frame_digest(frame), moves_used(frame) % 2, origin

    def priority(node, path, origin):
        frame = node.frame()
        avatar = avatar_cell(frame)
        prize = target(frame)
        distance = target_path_distance(frame)
        column = -1 if avatar is None else avatar[1]
        return (
            0 if prize is not None else 1,
            99 if distance is None else distance,
            -origin,
            -column,
            len(path),
        )

    frontier = [
        (priority(root, (), 0), next(counter), root, (), 0)
    ]
    seen = {key(root, 0)}
    evaluated = 0
    expanded = 0
    started = time.monotonic()
    best = priority(root, (), 0)
    best_path = ()
    max_origin = 0
    print(
        "TOP_FORWARD_ROOT", avatar_cell(root_frame), target(root_frame),
        tuple(controls(root_frame)), best, flush=True,
    )

    while frontier and evaluated < max_states:
        _score, _tie, node, path, origin = heapq.heappop(frontier)
        expanded += 1
        if len(path) >= max_depth:
            continue
        before = np.asarray(node.frame()).copy()
        local_effects = set()
        for action in action_choices(before):
            child = node.clone()
            child.step(*action)
            evaluated += 1
            child_path = (*path, action)
            if child.levels_completed > base_level:
                print(
                    "TOP_FORWARD_WIN", evaluated, expanded, child_path,
                    flush=True,
                )
                print(
                    "TOP_FORWARD_ROUTE",
                    [*decoded_route(), *child_path], flush=True,
                )
                return
            if child.terminal() or avatar_cell(child.frame()) is None:
                if evaluated >= max_states:
                    break
                continue
            child_frame = np.asarray(child.frame())
            child_origin = origin + signed_origin_delta(before, child_frame)
            child_key = key(child, child_origin)
            if child_key in local_effects:
                if evaluated >= max_states:
                    break
                continue
            local_effects.add(child_key)
            if child_key in seen:
                if evaluated >= max_states:
                    break
                continue
            seen.add(child_key)
            child_score = priority(child, child_path, child_origin)
            if child_score < best:
                best = child_score
                best_path = child_path
                print(
                    "TOP_FORWARD_PROGRESS", evaluated, expanded,
                    child_score, avatar_cell(child_frame),
                    target(child_frame), tuple(controls(child_frame)),
                    child_path, round(time.monotonic() - started, 1),
                    flush=True,
                )
            if child_origin > max_origin:
                max_origin = child_origin
                print(
                    "TOP_FORWARD_DESCENT", evaluated, expanded,
                    child_origin, avatar_cell(child_frame),
                    tuple(controls(child_frame)), child_path, flush=True,
                )
            heapq.heappush(
                frontier,
                (
                    child_score, next(counter), child,
                    child_path, child_origin,
                ),
            )
            if evaluated >= max_states:
                break
        if evaluated and evaluated % 100 < len(action_choices(before)):
            print(
                "TOP_FORWARD_SEARCH", evaluated, expanded,
                len(frontier), len(seen), best, max_origin,
                round(time.monotonic() - started, 1), flush=True,
            )
    print(
        "TOP_FORWARD_DONE", evaluated, expanded, len(frontier), len(seen),
        best, max_origin, best_path, flush=True,
    )


arena.run_program("bp35", probe)
