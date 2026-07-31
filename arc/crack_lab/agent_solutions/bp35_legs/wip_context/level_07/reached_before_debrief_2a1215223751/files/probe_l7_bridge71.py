"""Bounded local probe for the distinct upper chamber reached via support 7,1."""

from collections import deque
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action
from perception import connected_components, frame_delta
from probe_level7_reward_recovery import PREFIX, SUFFIX


LEFT, RIGHT, UNDO = (3,), (4,), (7,)
TOP_ROUTE = [
    *PREFIX,
    click_action(5, 2),
    *SUFFIX,
    LEFT, (6, 3, 9), RIGHT, (6, 3, 39),
    LEFT, LEFT, LEFT,
]
ENTRY = [
    (6, 3, 27), UNDO, click_action(7, 1), RIGHT, (6, 3, 21),
]


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
    return [
        (6, round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in connected_components(frame, colors=(8,), min_area=2)
        if blob.bbox[0] < 63 and blob.bbox[1] <= 5
    ]


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


def local_actions(node):
    frame = node.frame()
    avatar = avatar_cell(frame)
    if avatar is None:
        return []
    ai, aj = avatar
    out = [LEFT, RIGHT, UNDO, *controls(frame)]
    for i in range(max(0, ai - 4), min(10, ai + 5)):
        for j in range(max(0, aj - 3), min(8, aj + 4)):
            color, _ = _cell_shape(frame, i, j)
            if color in (0, 7, 12, 14, 15):
                out.append(click_action(i, j))
    return list(dict.fromkeys(out))


def selected_key(node):
    frame = np.asarray(node.frame())[:63].tobytes()
    undo = node.clone()
    undo.step(7)
    return frame, bool(undo.terminal()), np.asarray(undo.frame())[:63].tobytes()


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    for action in [*TOP_ROUTE, *ENTRY]:
        env.step(*action)
        if env.terminal():
            print("BRIDGE71_ENTRY_DEAD", action)
            return

    base_level = int(env.levels_completed)
    root = env.clone()
    before = np.asarray(root.frame()).copy()
    print(
        "BRIDGE71_ROOT", avatar_cell(before), target_cell(before),
        controls(before), lattice(before), flush=True,
    )
    for action in local_actions(root):
        child = root.clone()
        child.step(*action)
        delta = frame_delta(before[:63], child.frame()[:63])
        print(
            "BRIDGE71_ONE", action, int(child.levels_completed) - base_level,
            bool(child.terminal()),
            None if child.terminal() else avatar_cell(child.frame()),
            None if child.terminal() else target_cell(child.frame()),
            (delta["count"], delta["bbox"]),
            flush=True,
        )

    max_states = int(os.environ.get("MAX_STATES", "300"))
    max_depth = int(os.environ.get("MAX_DEPTH", "12"))
    queue = deque([(root, ())])
    seen = {selected_key(root)}
    observed = {
        (avatar_cell(root.frame()), target_cell(root.frame()), lattice(root.frame()))
    }
    expanded = 0
    while queue and expanded < max_states:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for action in local_actions(node):
            child = node.clone()
            child.step(*action)
            expanded += 1
            child_path = (*path, action)
            if child.levels_completed > base_level:
                print("BRIDGE71_WIN", child_path, expanded, flush=True)
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
                        "BRIDGE71_PROGRESS", expanded, child_path,
                        state[0], state[1], controls(child.frame()), state[2],
                        flush=True,
                    )
            key = selected_key(child)
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path))
            if expanded >= max_states:
                break
    print(
        "BRIDGE71_DONE", expanded, len(seen), len(queue), len(observed),
        flush=True,
    )


arena.run_program("bp35", probe)
