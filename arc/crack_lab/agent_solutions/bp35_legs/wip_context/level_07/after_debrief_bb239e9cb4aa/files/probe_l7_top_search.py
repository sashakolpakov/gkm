"""Selected-aware primitive search from a replay-verified level-7 top room."""

import heapq
import itertools
import json
import sys
import zlib

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
    (3,), (3,), (3,),
]
STATIC = {3: 1, 5: 1, 10: 2, 0: 3}


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


def choices(node):
    frame = node.frame()
    avatar = avatar_cell(frame)
    if avatar is None:
        return []
    ai, aj = avatar
    out = [(L, None), (R, None), ((7,), None)]
    controls = [
        blob for blob in connected_components(frame, colors=(8,), min_area=2)
        if blob.bbox[0] < 63 and blob.bbox[1] <= 5
    ]
    if controls:
        blob = min(
            controls,
            key=lambda item: abs(item.centroid[0] - ROW_ANCHORS[ai]),
        )
        y, x = blob.centroid
        out.append(((6, round(x), round(y)), "gravity"))
    frame_tag = zlib.crc32(np.asarray(frame)[:63, 6:].tobytes())
    for i, j in support_cells(frame):
        if abs(i - ai) <= 3 and abs(j - aj) <= 3:
            token = ("support", frame_tag, i, j)
            out.append(((6, COL_ANCHORS[j], ROW_ANCHORS[i]), token))
    return out


def frame_key(frame):
    return np.asarray(frame)[:63].tobytes()


def terrain_shift(before, after):
    scored = []
    for shift in range(-9, 10):
        matches = mismatches = 0
        for i, y in enumerate(ROW_ANCHORS):
            other = i + shift
            if not 0 <= other < 10:
                continue
            for x in COL_ANCHORS:
                left = STATIC.get(int(before[y][x]))
                right = STATIC.get(int(after[ROW_ANCHORS[other]][x]))
                if left is None or right is None:
                    continue
                matches += left == right
                mismatches += left != right
        scored.append((matches - 3 * mismatches, matches, -abs(shift), shift))
    shift = max(scored)[-1]
    return -shift


def dense(frame, descent):
    avatar, target = avatar_cell(frame), target_cell(frame)
    if avatar is None:
        return 0, 99, descent
    if target is None:
        return 0, 99, descent
    distance = abs(avatar[0] - target[0]) + abs(avatar[1] - target[1])
    return 1, distance, descent


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for action in TOP_ROUTE:
        env.step(*action)
        if env.terminal():
            print("TOP_ROUTE_FAILED", action)
            return

    base_level = int(env.levels_completed)
    root = env.clone()
    # The last selectable action in TOP_ROUTE is a gravity-strip click.
    root_selected = "gravity"
    tie = itertools.count()
    root_dense = dense(root.frame(), 0)
    frontier = [((0, 0, 0, 0), next(tie), root, (), root_selected, frozenset(), 0)]
    seen = {(frame_key(root.frame()), root_selected, frozenset())}
    best = root_dense
    expanded = 0
    print(
        "TOP_ROOT", avatar_cell(root.frame()), target_cell(root.frame()),
        support_cells(root.frame()), root_dense,
    )

    while frontier and expanded < 900:
        _, _, node, path, selected, toggled, descent = heapq.heappop(frontier)
        if len(path) >= 28:
            continue
        before = np.asarray(node.frame()).copy()
        for action, selection in choices(node):
            child = node.clone()
            child.step(*action)
            expanded += 1
            child_path = (*path, action)
            if child.levels_completed > base_level:
                print("TOP_WIN", child_path, expanded, flush=True)
                return
            if child.terminal() or avatar_cell(child.frame()) is None:
                continue

            child_selected = selected if selection is None else selection
            child_toggled = set(toggled)
            if selection not in (None, "gravity"):
                if selection in child_toggled:
                    child_toggled.remove(selection)
                else:
                    child_toggled.add(selection)
            elif action[0] == 7 and selected not in (None, "gravity"):
                if selected in child_toggled:
                    child_toggled.remove(selected)
                else:
                    child_toggled.add(selected)
            child_toggled = frozenset(child_toggled)
            child_descent = descent + terrain_shift(before, child.frame())
            progress = dense(child.frame(), child_descent)
            if (
                progress[0] > best[0]
                or progress[0] == best[0] and progress[1] < best[1]
                or progress[:2] == best[:2] and progress[2] > best[2]
            ):
                best = progress
                print(
                    "TOP_PROGRESS", best, len(child_path),
                    avatar_cell(child.frame()), target_cell(child.frame()),
                    child_path, flush=True,
                )
            key = (frame_key(child.frame()), child_selected, child_toggled)
            if key in seen:
                continue
            seen.add(key)
            target_seen, target_distance, down = progress
            priority = (
                -target_seen,
                target_distance,
                -down,
                len(child_path),
            )
            heapq.heappush(
                frontier,
                (priority, next(tie), child, child_path,
                 child_selected, child_toggled, child_descent),
            )
            if expanded >= 900:
                break
        if expanded and expanded % 100 < len(choices(node)):
            print("TOP_SEARCH", expanded, len(frontier), len(seen), best, flush=True)
    print("TOP_DONE", expanded, len(frontier), len(seen), best)


levels, path, error = arena.run_program("bp35", probe)
print("RESULT", levels, len(path), error)
