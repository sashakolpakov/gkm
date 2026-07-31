"""Selected-support-aware search from the decoded post-switch side room."""

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
    avatar = avatar_cell(frame)
    if avatar is None:
        return []
    ai, aj = avatar
    out = [(L, None), (R, None), ((7,), None)]
    visible = controls(frame)
    if visible:
        blob = min(visible, key=lambda item: abs(item.centroid[0] - ROW_ANCHORS[ai]))
        y, x = blob.centroid
        # Gravity-strip clicks do not replace the remotely selected support.
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
    for action in [*decoded_route(), L, L, L, L]:
        env.step(*action)
    switches = controls(env.frame())
    if switches:
        blob = max(switches, key=lambda item: item.centroid[0])
        y, x = blob.centroid
        env.step(6, round(x), round(y))

    base_level = int(env.levels_completed)
    root = env.clone()
    root_selected = ("root_support",)
    queue = deque([(root, (), root_selected)])
    seen = {(np.asarray(root.frame())[:63].tobytes(), root_selected)}
    expanded = 0
    print(
        "FINAL_ROOT", avatar_cell(root.frame()), target_cell(root.frame()),
        support_cells(root.frame()), len(controls(root.frame())), flush=True,
    )
    while queue and expanded < 500:
        node, path, selected = queue.popleft()
        if len(path) >= 14:
            continue
        for action, selection in choices(node):
            child = node.clone()
            child.step(*action)
            expanded += 1
            child_path = (*path, action)
            if child.levels_completed > base_level:
                print("FINAL_WIN", child_path, expanded, flush=True)
                return
            if child.terminal() or avatar_cell(child.frame()) is None:
                continue
            target = target_cell(child.frame())
            if target is not None:
                print(
                    "FINAL_TARGET", child_path, expanded,
                    avatar_cell(child.frame()), target, flush=True,
                )
            child_selected = selected if selection is None else selection
            key = (np.asarray(child.frame())[:63].tobytes(), child_selected)
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path, child_selected))
            if expanded >= 500:
                break
        if expanded and expanded % 100 < len(choices(node)):
            print("FINAL_SEARCH", expanded, len(queue), len(seen), flush=True)
    print("FINAL_DONE", expanded, len(queue), len(seen))


levels, path, error = arena.run_program("bp35", probe)
print("RESULT", levels, len(path), error)
