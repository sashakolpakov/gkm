"""Bounded local search for the level-7 hazard-pocket exit."""

from collections import deque
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS, click_action
from perception import connected_components
from probe_level7_reward_recovery import PREFIX, SUFFIX


L, R = (3,), (4,)
TOP_ROUTE = [
    *PREFIX,
    click_action(5, 2),
    *SUFFIX,
    (3,), (6, 3, 9), (4,), (6, 3, 39),
    L, L, L,
]
BRIDGE72 = [
    (6, 3, 27), (7,), click_action(7, 2), R, (6, 3, 21),
    R, R, R, R,
]
POCKET_ROUTE = [
    *BRIDGE72,
    (6, 3, 0),
    click_action(6, 3),
    R,
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


def support_cells(frame):
    return [
        (i, j)
        for i, y in enumerate(ROW_ANCHORS)
        for j, x in enumerate(COL_ANCHORS)
        if int(frame[y][x]) in (12, 14)
    ]


def choices(node):
    frame = node.frame()
    avatar = avatar_cell(frame)
    if avatar is None:
        return []
    ai, aj = avatar
    out = [(L, None), (R, None), ((7,), None)]
    for i, j in support_cells(frame):
        if abs(i - ai) <= 3 and abs(j - aj) <= 3:
            out.append((click_action(i, j), ("support", i, j)))
    controls = [
        blob for blob in connected_components(frame, colors=(8,), min_area=2)
        if blob.bbox[0] < 63 and blob.bbox[1] <= 5
    ]
    if controls:
        blob = min(controls, key=lambda item: abs(item.centroid[0] - ROW_ANCHORS[ai]))
        y, x = blob.centroid
        out.append(((6, round(x), round(y)), ("gravity",)))
    return list(dict.fromkeys(out))


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for action in [*TOP_ROUTE, *POCKET_ROUTE]:
        env.step(*action)
        if env.terminal():
            print("POCKET_ROOT_FAILED", action)
            return

    base_level = int(env.levels_completed)
    root = env.clone()
    root_avatar = avatar_cell(root.frame())
    print("POCKET_ROOT", root_avatar, support_cells(root.frame()))
    queue = deque([(root, (), ("support", 6, 3))])
    seen = {(np.asarray(root.frame())[:63].tobytes(), ("support", 6, 3))}
    expanded = 0
    max_depth = 9
    while queue and expanded < 600:
        node, path, selected = queue.popleft()
        if len(path) >= max_depth:
            continue
        for action, selection in choices(node):
            child = node.clone()
            child.step(*action)
            expanded += 1
            child_path = (*path, action)
            avatar = None if child.terminal() else avatar_cell(child.frame())
            if child.levels_completed > base_level:
                print("POCKET_WIN", child_path, expanded)
                return
            if not child.terminal() and avatar is not None and avatar[1] >= 4:
                print("POCKET_EXIT", child_path, expanded, avatar)
                return
            if child.terminal() or avatar is None:
                continue
            child_selected = selected if selection is None else selection
            key = (np.asarray(child.frame())[:63].tobytes(), child_selected)
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path, child_selected))
            if expanded >= 600:
                break
        if expanded and expanded % 100 < len(choices(node)):
            print("POCKET_SEARCH", expanded, len(queue), len(seen), flush=True)
    print("POCKET_DONE", expanded, len(queue), len(seen))


levels, path, error = arena.run_program("bp35", probe)
print("RESULT", levels, len(path), error)
