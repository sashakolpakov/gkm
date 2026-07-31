"""History-aware replay search for level 7's rewind/gravity rooms."""

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
import gkm_legs as campaign

from legs import COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action, run_actions
from perception import connected_components
from probe_l7_fresh_graph import signed_origin_delta
from probe_l7_raw_search import SEED, avatar_position, target_path_distance


if campaign._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


LEFT, RIGHT, UNDO = (3,), (4,), (7,)
ROOT_ROUTE = [*SEED, UNDO, UNDO]


def digest(frame):
    return hashlib.blake2b(
        np.asarray(frame)[:63].tobytes(), digest_size=8
    ).digest()


def history_push(history, action, frame):
    if action[0] == 7:
        return history[:-1] if history else history
    return (*history, (tuple(action), digest(frame)))


def controls(frame, ay):
    blobs = [
        blob for blob in connected_components(frame, colors=(8,), min_area=3)
        if blob.bbox[1] <= 5 and blob.bbox[0] < 63
    ]
    if not blobs:
        return []
    ordered = [
        min(blobs, key=lambda blob: abs(blob.centroid[0] - ay)),
        min(blobs, key=lambda blob: blob.centroid[0]),
        max(blobs, key=lambda blob: blob.centroid[0]),
    ]
    out = []
    for blob in ordered:
        y, x = blob.centroid
        action = (6, int(round(x)), int(round(y)))
        if action not in out:
            out.append(action)
    return out


def choices(frame, history):
    avatar = avatar_position(frame)
    if avatar is None:
        return []
    ax, ay = avatar
    ai = min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - ay))
    aj = min(range(8), key=lambda j: abs(COL_ANCHORS[j] - ax))
    out = [LEFT, RIGHT]
    if history:
        out.append(UNDO)
    out.extend(controls(frame, ay))
    row_radius = int(os.environ.get("ROW_RADIUS", "4"))
    col_radius = int(os.environ.get("COL_RADIUS", "2"))
    for i in range(max(0, ai - row_radius), min(10, ai + row_radius + 1)):
        for j in range(max(0, aj - col_radius), min(8, aj + col_radius + 1)):
            color, _area = _cell_shape(frame, i, j)
            if color in (12, 14):
                out.append(click_action(i, j))
            elif color == 15 and abs(i - ai) <= 1 and abs(j - aj) <= 1:
                out.append(click_action(i, j))
    if os.environ.get("INCLUDE_WAIT") == "1":
        out.append((6, 9, 3))
    return list(dict.fromkeys(out))


def expanded_supports(frame):
    return sum(
        _cell_shape(frame, i, j)[0] == 12
        and _cell_shape(frame, i, j)[1] >= 13
        for i in range(10)
        for j in range(8)
    )


def metrics(node, origin):
    frame = node.frame()
    avatar = avatar_position(frame)
    if avatar is None:
        return None
    ai = min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - avatar[1]))
    aj = min(range(8), key=lambda j: abs(COL_ANCHORS[j] - avatar[0]))
    distance = target_path_distance(frame)
    target = bool(np.any(np.asarray(frame)[:63] == 7))
    return {
        "avatar": (ai, aj),
        "origin": origin,
        "world_row": origin + ai,
        "distance": distance,
        "target": target,
        "expanded": expanded_supports(frame),
    }


def priority(path, data):
    row, col = data["avatar"]
    distance = data["distance"]
    if distance is not None and distance < 18:
        phase = 0
    elif data["target"] and col >= 6:
        phase = 1
    elif col >= 6:
        phase = 2
    elif data["target"]:
        phase = 3
    else:
        phase = 4
    return (
        phase,
        30 if distance is None else distance,
        -data["expanded"],
        -abs(data["origin"]),
        len(path),
        abs(6 - col),
        row,
    )


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    history = ()
    origin = 0
    for action in ROOT_ROUTE:
        before = np.asarray(env.frame()).copy()
        history = history_push(history, action, before)
        env.step(*action)
        if env.terminal():
            print("STACK_ROOT_TERMINAL", action)
            return
        origin += signed_origin_delta(before, env.frame())
    base_level = int(env.levels_completed)
    root = env.clone()
    root_data = metrics(root, origin)
    print(
        "STACK_ROOT", len(ROOT_ROUTE), len(history), root_data,
        flush=True,
    )

    max_states = int(os.environ.get("MAX_STATES", "4000"))
    max_depth = int(os.environ.get("MAX_DEPTH", "36"))
    counter = itertools.count()
    queue = [
        (
            priority((), root_data), next(counter), (),
            history, origin, root_data,
        )
    ]
    seen = {(digest(root.frame()), history)}
    started = time.monotonic()
    best = priority((), root_data)
    expanded = generated = 0
    while queue and expanded < max_states:
        _, _, path, node_history, node_origin, data = heapq.heappop(queue)
        node = root.clone()
        run_actions(node, path)
        expanded += 1
        if len(path) >= max_depth or node.terminal():
            continue
        before = np.asarray(node.frame()).copy()
        for action in choices(before, node_history):
            child = node.clone()
            child.step(*action)
            generated += 1
            child_path = (*path, action)
            if child.levels_completed > base_level:
                route = [*ROOT_ROUTE, *child_path]
                print(
                    "STACK_WIN", expanded, generated, len(route), route,
                    round(time.monotonic() - started, 2), flush=True,
                )
                return
            if child.terminal() or avatar_position(child.frame()) is None:
                continue
            after = np.asarray(child.frame())
            child_history = history_push(node_history, action, before)
            child_origin = node_origin + signed_origin_delta(before, after)
            key = (digest(after), child_history)
            if key in seen:
                continue
            seen.add(key)
            child_data = metrics(child, child_origin)
            child_priority = priority(child_path, child_data)
            if child_priority < best:
                best = child_priority
                print(
                    "STACK_PROGRESS", expanded, generated, child_priority,
                    child_data, len(child_history), child_path,
                    round(time.monotonic() - started, 2), flush=True,
                )
            heapq.heappush(
                queue,
                (
                    child_priority, next(counter), child_path,
                    child_history, child_origin, child_data,
                ),
            )
        if expanded % 100 == 0:
            print(
                "STACK_SEARCH", expanded, generated, len(queue), len(seen),
                best, len(path), round(time.monotonic() - started, 2),
                flush=True,
            )
    print(
        "STACK_DONE", expanded, generated, len(queue), len(seen), best,
        round(time.monotonic() - started, 2), flush=True,
    )


arena.run_program("bp35", probe)
