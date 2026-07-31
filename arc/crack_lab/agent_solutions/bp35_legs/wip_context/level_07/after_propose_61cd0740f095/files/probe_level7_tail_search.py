import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action, moves_used,
)
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


PREFIX = [
    (4,), (4,), (4,), click_action(8, 4), (6, 3, 3),
    (4,), (6, 3, 3), (4,),
    (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
    (3,), (4,), (4,), click_action(8, 2), (6, 3, 9),
    (3,), (3,), (6, 3, 15), (3,), (3,),
    (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
]


def avatar_cell(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - int(ys.mean()))),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - int(xs.mean()))),
    )


def gravity_action(frame):
    return next(
        ((6, 3, y) for y in ROW_ANCHORS if int(frame[y][3]) == 8),
        None,
    )


def run(node, route):
    actions = []
    for token in route:
        action = gravity_action(node.frame()) if token == "g" else token
        if action is None or node.terminal():
            break
        node.step(*action)
        actions.append(action)
    return actions


def choices(frame):
    avatar = avatar_cell(frame)
    out = [(3,), (4,)]
    if avatar is None:
        return out
    ai, _ = avatar
    for i in range(max(0, ai - 2), min(10, ai + 4)):
        for j in range(8):
            color, _ = _cell_shape(frame, i, j)
            if color in (12, 14):
                out.append(click_action(i, j))
    return out


def key(node):
    frame = arr(node.frame())
    return frame[:63].tobytes(), moves_used(frame) % 2


def search(root, max_states=300, max_depth=9):
    queue = deque([(root.clone(), ())])
    seen = {key(root)}
    expanded = 0
    while queue and expanded < max_states:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for action in choices(node.frame()):
            child = node.clone()
            child.step(*action)
            expanded += 1
            child_path = path + (action,)
            if child.levels_completed > 6:
                return child_path, expanded, "reward"
            if child.terminal() or avatar_cell(child.frame()) is None:
                continue
            if gravity_action(child.frame()) is not None:
                return child_path, expanded, "control"
            child_key = key(child)
            if child_key not in seen:
                seen.add(child_key)
                queue.append((child, child_path))
            if expanded >= max_states:
                break
    return (), expanded, "none"


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    node = env.clone()
    run(node, PREFIX)
    for _ in range(2):
        run(node, [(3,), "g", (4,), "g"])
    route, expanded, reason = search(node)
    verified = node.clone()
    run(verified, route)
    print("SEARCH", expanded, reason, list(route))
    print(
        "VERIFY", verified.levels_completed, verified.terminal(),
        avatar_cell(verified.frame()), gravity_action(verified.frame()),
    )


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
