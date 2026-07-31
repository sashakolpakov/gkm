"""Bounded direct-clone search over verified level-9 affordances."""

import heapq
import itertools
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import solve
from legs import (
    AVATAR_COLORS,
    COL_ANCHORS,
    ROW_ANCHORS,
    WALL_COLORS,
    _cell_shape,
    moves_used,
)
from perception import connected_components


MAX_STATES = 5000
MAX_DEPTH = 70
MAX_SECONDS = 45.0


def enter_level_9(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    solve.solve(env)


def avatar_cell(frame):
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            if int(frame[y][x]) in AVATAR_COLORS:
                return i, j
    return None


def control_actions(frame):
    rows = [row for row in range(63) if int(frame[row][3]) == 8]
    runs = []
    for row in rows:
        if not runs or row != runs[-1][-1] + 1:
            runs.append([row])
        else:
            runs[-1].append(row)
    return [(6, 3, run[len(run) // 2]) for run in runs]


def actions(node):
    frame = node.frame()
    avatar = avatar_cell(frame)
    out = [(3,), (4,), *control_actions(frame)]
    if avatar is None:
        return out
    ai, aj = avatar
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            color, area = _cell_shape(frame, i, j)
            if color == 14:
                out.append((6, x, y))
            elif color in (12, 15) and area >= 8 and abs(i - ai) + abs(j - aj) <= 1:
                out.append((6, x, y))
    return list(dict.fromkeys(out))


def physical_frame(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return frame


def state_key(env, height):
    return height, moves_used(env.frame()) % 2, physical_frame(env).tobytes()


def signed_shift(before, after):
    a = np.asarray(before)[:63]
    b = np.asarray(after)[:63]
    scored = []
    for bands in range(-9, 10):
        offset = 6 * abs(bands)
        if bands >= 0:
            left, right = a[:63 - offset], b[offset:]
        else:
            left, right = a[offset:], b[:63 - offset]
        hits = int(np.all(left == right, axis=1).sum())
        scored.append((hits, -abs(bands), bands))
    return max(scored)[2]


def reachable_goal_bonus(frame):
    avatar = avatar_cell(frame)
    goals = [
        blob for blob in connected_components(frame, colors=(7,), min_area=3)
        if blob.bbox[0] < 63
    ]
    if avatar is None or not goals:
        return 0
    ai, aj = avatar
    goal_cells = {
        (
            min(range(len(ROW_ANCHORS)), key=lambda i: abs(ROW_ANCHORS[i] - blob.centroid[0])),
            min(range(len(COL_ANCHORS)), key=lambda j: abs(COL_ANCHORS[j] - blob.centroid[1])),
        )
        for blob in goals
    }
    queue = [(ai, aj)]
    seen = {(ai, aj)}
    while queue:
        cell = queue.pop()
        if cell in goal_cells:
            return 2
        i, j = cell
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if not (0 <= ni < len(ROW_ANCHORS) and 0 <= nj < len(COL_ANCHORS)):
                continue
            if (ni, nj) in seen:
                continue
            color = int(frame[ROW_ANCHORS[ni]][COL_ANCHORS[nj]])
            if color in WALL_COLORS:
                continue
            seen.add((ni, nj))
            queue.append((ni, nj))
    return 1


def search(root):
    base_level = int(root.levels_completed)
    start = root.clone()
    counter = itertools.count()
    queue = [(0, 0, 0, next(counter), start, (), 0, 0)]
    seen = {state_key(start, 0)}
    started = time.monotonic()
    expanded = 0
    best = (0, 0, ())
    while queue and expanded < MAX_STATES and time.monotonic() - started < MAX_SECONDS:
        _, _, _, _, node, path, height, peak = heapq.heappop(queue)
        expanded += 1
        if peak > best[0] or (peak == best[0] and height > best[1]):
            best = (peak, height, path)
        if expanded % 250 == 0:
            print(
                "PROGRESS",
                {"expanded": expanded, "seen": len(seen), "depth": len(path),
                 "height": height, "peak": peak, "queue": len(queue)},
                flush=True,
            )
        if len(path) >= MAX_DEPTH:
            continue
        parent_frame = np.asarray(node.frame()).copy()
        parent_physical = physical_frame(node)
        for action in actions(node):
            child = node.clone()
            child.step(*action)
            child_path = path + (action,)
            if int(child.levels_completed) > base_level:
                return list(child_path), expanded, len(seen), best
            if child.terminal() or avatar_cell(child.frame()) is None:
                continue
            child_physical = physical_frame(child)
            if np.array_equal(parent_physical, child_physical):
                continue
            next_height = height + signed_shift(parent_frame, child.frame())
            next_peak = max(peak, next_height)
            key = state_key(child, next_height)
            if key in seen:
                continue
            seen.add(key)
            bonus = reachable_goal_bonus(child.frame())
            heapq.heappush(
                queue,
                (
                    -bonus,
                    -next_peak,
                    len(child_path),
                    next(counter),
                    child,
                    child_path,
                    next_height,
                    next_peak,
                ),
            )
    return [], expanded, len(seen), best


def probe(env):
    enter_level_9(env)
    path, expanded, seen, best = search(env)
    print(
        "SEARCH",
        {"found": bool(path), "length": len(path), "expanded": expanded,
         "seen": seen, "best_peak": best[0], "best_height": best[1]},
    )
    print("PATH", path)
    print("BEST_PATH", list(best[2]))
    if path:
        child = env.clone()
        for action in path:
            child.step(*action)
        print(
            "VERIFY",
            {"level": int(child.levels_completed) + 1,
             "terminal": bool(child.terminal()), "moves": moves_used(child.frame())},
        )


arena.run_program("bp35", probe)
