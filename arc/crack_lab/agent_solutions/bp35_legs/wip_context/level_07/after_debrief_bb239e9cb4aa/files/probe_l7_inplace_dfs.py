"""History-aware DFS using action 7 to backtrack one clone in place."""

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action, run_actions
from perception import connected_components
from probe_l7_fresh_graph import signed_origin_delta
from probe_l7_raw_search import SEED, avatar_position, target_path_distance


LEFT, RIGHT, RELEASE = (3,), (4,), (7,)


def stack_after(actions):
    stack = []
    for action in actions:
        action = tuple(action)
        if action[0] == 7:
            if stack:
                stack.pop()
        else:
            stack.append(action)
    return tuple(stack)


def digest(frame):
    return hashlib.blake2b(
        np.asarray(frame)[:63].tobytes(), digest_size=12
    ).digest()


def avatar_cell(frame):
    avatar = avatar_position(frame)
    if avatar is None:
        return None
    x, y = avatar
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - y)),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - x)),
    )


def controls(frame):
    out = []
    for blob in connected_components(frame, colors=(8,), min_area=1):
        if blob.bbox[0] >= 63 or blob.bbox[1] > 5:
            continue
        y, x = blob.centroid
        action = (6, round(x), round(y))
        if action not in out:
            out.append(action)
    return out


def choices(frame):
    avatar = avatar_cell(frame)
    if avatar is None:
        return []
    ai, aj = avatar
    out = [RIGHT, LEFT, RELEASE, *controls(frame)]
    for i in range(max(0, ai - 4), min(10, ai + 5)):
        for j in range(max(0, aj - 3), min(8, aj + 4)):
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


def metric(frame, origin, depth):
    cell = avatar_cell(frame)
    if cell is None:
        return (9, 99, 99, 99, depth)
    distance = target_path_distance(frame)
    target_seen = distance is not None
    row, col = cell
    if distance is not None and distance < 18:
        phase = 0
    elif target_seen and col >= 6:
        phase = 1
    elif target_seen:
        phase = 2
    elif col >= 6:
        phase = 3
    else:
        phase = 4
    return (
        phase,
        99 if distance is None else distance,
        -expanded_supports(frame),
        -abs(origin),
        -col,
        row,
        depth,
    )


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    base_level = int(env.levels_completed)
    root_actions = [*SEED, RELEASE, RELEASE]
    for action in root_actions:
        env.step(*action)
        if env.terminal():
            print("INPLACE_ROOT_DEAD", action)
            return
    root = env.clone()
    node = root.clone()
    root_stack = stack_after(root_actions)
    root_origin = 0
    max_states = int(os.environ.get("MAX_STATES", "900"))
    max_depth = int(os.environ.get("MAX_DEPTH", "20"))
    seen = {(digest(node.frame()), root_stack)}
    expanded = 0
    env_steps = 0
    best = metric(node.frame(), root_origin, 0)
    best_path = ()
    print(
        "INPLACE_ROOT", avatar_cell(node.frame()),
        target_path_distance(node.frame()), len(root_stack), best, flush=True,
    )

    def reconstruct(path):
        fresh = root.clone()
        run_actions(fresh, path)
        return fresh

    def restore(path, before_key, action, popped):
        nonlocal node, env_steps
        inverse = popped if action[0] == 7 else RELEASE
        if inverse is None:
            node = reconstruct(path)
            return
        node.step(*inverse)
        env_steps += 1
        if node.terminal() or digest(node.frame()) != before_key:
            node = reconstruct(path)

    def dfs(path, stack, origin):
        nonlocal node, expanded, env_steps, best, best_path
        if expanded >= max_states or len(path) >= max_depth:
            return None
        if env_steps and env_steps % 260 == 0:
            node = reconstruct(path)

        frame = np.asarray(node.frame()).copy()
        before_key = digest(frame)
        ranked = []
        for action in choices(frame):
            popped = stack[-1] if action[0] == 7 and stack else None
            if action[0] == 7 and popped is None:
                continue
            node.step(*action)
            env_steps += 1
            if node.levels_completed > base_level:
                return (*path, action)
            if node.terminal() or avatar_cell(node.frame()) is None:
                node = reconstruct(path)
                continue
            child_frame = np.asarray(node.frame())
            child_origin = origin + signed_origin_delta(frame, child_frame)
            child_stack = (
                stack[:-1] if action[0] == 7 else (*stack, action)
            )
            child_key = (digest(child_frame), child_stack)
            child_metric = metric(child_frame, child_origin, len(path) + 1)
            restore(path, before_key, action, popped)
            if child_key in seen:
                continue
            ranked.append(
                (
                    child_metric, action, popped, child_origin,
                    child_stack, child_key,
                )
            )

        ranked.sort(key=lambda item: item[0])
        for child_metric, action, popped, child_origin, child_stack, child_key in ranked:
            if expanded >= max_states:
                break
            node.step(*action)
            env_steps += 1
            if node.levels_completed > base_level:
                return (*path, action)
            if node.terminal() or avatar_cell(node.frame()) is None:
                node = reconstruct(path)
                continue
            expanded += 1
            child_path = (*path, action)
            seen.add(child_key)
            if child_metric < best:
                best = child_metric
                best_path = child_path
                print(
                    "INPLACE_PROGRESS", expanded, env_steps, best,
                    avatar_cell(node.frame()),
                    target_path_distance(node.frame()), child_path,
                    flush=True,
                )
            result = dfs(child_path, child_stack, child_origin)
            if result is not None:
                return result
            restore(path, before_key, action, popped)
        return None

    winner = dfs((), root_stack, root_origin)
    if winner is not None:
        verified = root.clone()
        run_actions(verified, winner)
        print(
            "INPLACE_WIN", winner, int(verified.levels_completed),
            bool(verified.terminal()), expanded, env_steps, flush=True,
        )
    else:
        print(
            "INPLACE_DONE", expanded, env_steps, len(seen),
            best, best_path, flush=True,
        )


arena.run_program("bp35", probe)
