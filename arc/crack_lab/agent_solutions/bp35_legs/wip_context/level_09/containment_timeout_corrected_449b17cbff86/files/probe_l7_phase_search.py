"""Bounded clone search from the isolated phase-changing insertion."""

from collections import deque
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS, click_action
from perception import connected_components
from probe_level7_decoded_stage import decoded_route


L, R = (3,), (4,)
INSERT_BOUNDARY = 48
INSERT = (6, 27, 39)


def cell(frame, colour):
    ys, xs = np.where(np.asarray(frame)[:63] == colour)
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


def controls(frame):
    return [
        blob for blob in connected_components(frame, colors=(8,), min_area=2)
        if blob.bbox[0] < 63 and blob.bbox[1] <= 5
    ]


def choices(node):
    frame = node.frame()
    avatar = cell(frame, 9)
    if avatar is None:
        return []
    ai, aj = avatar
    out = [((3,), None), ((4,), None), ((7,), None)]
    for blob in controls(frame):
        y, x = blob.centroid
        out.append(((6, round(x), round(y)), None))
    for i, j in support_cells(frame):
        if abs(i - ai) <= 3 and abs(j - aj) <= 3:
            out.append((click_action(i, j), ("support", i, j)))
    return list(dict.fromkeys(out))


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    route = [*decoded_route(), (3,), (3,), (3,), (3,)]
    for index, action in enumerate(route):
        if index == INSERT_BOUNDARY:
            env.step(*INSERT)
        env.step(*action)
    switches = controls(env.frame())
    if switches:
        blob = max(switches, key=lambda item: item.centroid[0])
        y, x = blob.centroid
        env.step(6, round(x), round(y))

    base_level = int(env.levels_completed)
    root = env.clone()
    selected = ("inserted", INSERT)
    queue = deque([(root, (), selected)])
    seen = {(np.asarray(root.frame())[:63].tobytes(), selected)}
    expanded = 0
    print(
        "PHASE_ROOT", cell(root.frame(), 9), cell(root.frame(), 7),
        len(controls(root.frame())), flush=True,
    )
    while queue and expanded < 600:
        node, path, selected = queue.popleft()
        if len(path) >= 16:
            continue
        options = choices(node)
        for action, selection in options:
            child = node.clone()
            child.step(*action)
            expanded += 1
            child_path = (*path, action)
            if child.levels_completed > base_level:
                print("PHASE_WIN", child_path, expanded, flush=True)
                return
            avatar = None if child.terminal() else cell(child.frame(), 9)
            if avatar is None:
                continue
            target = cell(child.frame(), 7)
            if target is not None:
                print("PHASE_TARGET", child_path, avatar, target, flush=True)
            child_selected = selected if selection is None else selection
            key = (np.asarray(child.frame())[:63].tobytes(), child_selected)
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path, child_selected))
            if expanded >= 600:
                break
        if expanded and expanded % 100 < len(options):
            print("PHASE_SEARCH", expanded, len(queue), len(seen), flush=True)
    print("PHASE_DONE", expanded, len(queue), len(seen), flush=True)


levels, path, error = arena.run_program("bp35", probe)
print("RESULT", levels, len(path), error)
