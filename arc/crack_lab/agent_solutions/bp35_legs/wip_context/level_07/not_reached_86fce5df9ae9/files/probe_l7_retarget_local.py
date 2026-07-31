"""Bounded local search in the lower-control retargeted side chamber."""

from collections import deque
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action, run_actions
from perception import connected_components
from probe_level7_decoded_stage import decoded_route


LEFT, RIGHT, UNDO = (3,), (4,), (7,)


def avatar_cell(frame):
    ys, xs = np.where(np.asarray(frame) == 9)
    if not len(xs):
        return None
    y, x = float(ys.mean()), float(xs.mean())
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - y)),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - x)),
    )


def target_cell(frame):
    ys, xs = np.where(np.asarray(frame)[:63] == 7)
    if not len(xs):
        return None
    y, x = float(ys.mean()), float(xs.mean())
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


def lattice(frame):
    palette = {
        0: "v", 3: "#", 5: "#", 7: "T", 8: "g", 9: "A",
        10: ".", 11: "a", 12: "s", 14: "Y", 15: "h",
    }
    return "/".join(
        "".join(
            palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS
        )
        for y in ROW_ANCHORS
    )


def actions(node):
    frame = node.frame()
    avatar = avatar_cell(frame)
    if avatar is None:
        return []
    ai, aj = avatar
    out = [LEFT, RIGHT, UNDO, *controls(frame)]
    for i in range(max(0, ai - 4), min(10, ai + 5)):
        for j in range(max(0, aj - 4), min(8, aj + 5)):
            color, _area = _cell_shape(frame, i, j)
            if color in (12, 14, 15):
                out.append(click_action(i, j))
    return list(dict.fromkeys(out))


def stack_key(path):
    stack = []
    for action in path:
        if action[0] == 7:
            if stack:
                stack.pop()
        else:
            stack.append(action)
    return len(stack), tuple(stack[-8:])


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    base_level = int(env.levels_completed)

    repaired_route = []
    repaired = False
    for action in decoded_route():
        candidate = action
        if (
            not repaired
            and len(action) == 3
            and action[0] == 6
            and action[1] <= 5
            and int(env.frame()[action[2]][action[1]]) != 8
        ):
            visible = controls(env.frame())
            if visible:
                candidate = max(visible, key=lambda item: item[2])
                repaired = True
        env.step(*candidate)
        repaired_route.append(candidate)
        if env.terminal():
            print("RETARGET_LOCAL_ENTRY_DEAD", candidate)
            return
    run_actions(env, [LEFT] * 4)
    root = env.clone()
    print(
        "RETARGET_LOCAL_ROOT", avatar_cell(root.frame()),
        target_cell(root.frame()), controls(root.frame()),
        lattice(root.frame()), flush=True,
    )

    max_states = int(os.environ.get("MAX_STATES", "240"))
    max_depth = int(os.environ.get("MAX_DEPTH", "12"))
    queue = deque([()])
    seen = {
        (
            np.asarray(root.frame())[:63].tobytes(),
            stack_key(()),
        )
    }
    observed = {
        (avatar_cell(root.frame()), target_cell(root.frame()), lattice(root.frame()))
    }
    expanded = 0
    while queue and expanded < max_states:
        path = queue.popleft()
        if len(path) >= max_depth:
            continue
        node = root.clone()
        run_actions(node, path)
        for action in actions(node):
            child = node.clone()
            child.step(*action)
            expanded += 1
            child_path = (*path, action)
            if child.levels_completed > base_level:
                print(
                    "RETARGET_LOCAL_WIN", child_path,
                    [*repaired_route, *([LEFT] * 4), *child_path],
                    expanded, flush=True,
                )
                return
            if child.terminal() or avatar_cell(child.frame()) is None:
                if expanded >= max_states:
                    break
                continue
            state = (
                avatar_cell(child.frame()),
                target_cell(child.frame()),
                lattice(child.frame()),
            )
            if state not in observed:
                observed.add(state)
                if state[1] is not None or state[0] != avatar_cell(root.frame()):
                    print(
                        "RETARGET_LOCAL_PROGRESS", expanded, child_path,
                        state[0], state[1], controls(child.frame()), state[2],
                        flush=True,
                    )
            key = (
                np.asarray(child.frame())[:63].tobytes(),
                stack_key(child_path),
            )
            if key not in seen:
                seen.add(key)
                queue.append(child_path)
            if expanded >= max_states:
                break
    print(
        "RETARGET_LOCAL_DONE", expanded, len(seen), len(queue),
        len(observed), flush=True,
    )


arena.run_program("bp35", probe)
