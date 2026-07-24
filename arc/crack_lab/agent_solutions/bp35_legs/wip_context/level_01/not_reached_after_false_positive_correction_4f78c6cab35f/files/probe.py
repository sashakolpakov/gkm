"""Compact observational probes for the current raw-frame level."""

import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import color_counts, connected_components, frame_delta


ACTIONS = (3, 4, 6, 7)


def blobs(frame, min_area=4):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=min_area)
    ]


def changed_counts(before, after):
    a = color_counts(before)
    b = color_counts(after)
    return {
        color: b.get(color, 0) - a.get(color, 0)
        for color in sorted(set(a) | set(b))
        if a.get(color, 0) != b.get(color, 0)
    }


def probe(env):
    base = env.frame()
    print("actions", list(env.actions))
    print("level", env.levels_completed, "counts", color_counts(base))
    print("blobs", blobs(base))
    for action in ACTIONS:
        clone = env.clone()
        clone.step(action)
        print(
            "action",
            action,
            "reward",
            clone.levels_completed,
            "delta",
            frame_delta(base, clone.frame()),
            "count_delta",
            changed_counts(base, clone.frame()),
            "blobs",
            blobs(clone.frame()),
        )


def row_runs(row):
    runs = []
    start = 0
    for col in range(1, len(row) + 1):
        if col == len(row) or int(row[col]) != int(row[start]):
            runs.append((start, col - 1, int(row[start])))
            start = col
    return tuple(runs)


def overview(env):
    frame = env.frame()
    patterns = []
    start = 0
    last = row_runs(frame[0])
    for row in range(1, 63):
        current = row_runs(frame[row])
        if current != last:
            patterns.append((start, row - 1, last))
            start, last = row, current
    patterns.append((start, 62, last))
    print("row_patterns")
    for item in patterns:
        print(item)
    print("small_blobs", blobs(frame, min_area=1))


def click_probe(env):
    base = env.frame()
    points = [
        (0, 0),
        (15, 3),
        (15, 15),
        (21, 39),
        (33, 15),
        (39, 21),
        (58, 58),
    ]
    for x, y in points:
        clone = env.clone()
        clone.step(6, x, y)
        print(
            "click",
            (x, y),
            "reward",
            clone.levels_completed,
            "delta",
            frame_delta(base, clone.frame()),
            "count_delta",
            changed_counts(base, clone.frame()),
        )


def clear_probe(env):
    clone = env.clone()
    centers = [(15, 3), (21, 3), (27, 3), (33, 15), (39, 15), (45, 15), (51, 15)]
    for index, (x, y) in enumerate(centers, 1):
        before = clone.frame()
        clone.step(6, x, y)
        remaining = [
            (b.bbox, b.area)
            for b in connected_components(clone.frame(), colors=(14,), min_area=1)
        ]
        print(
            "clear",
            index,
            (x, y),
            "changed",
            frame_delta(before, clone.frame())["count"],
            "remaining",
            remaining,
            "reward",
            clone.levels_completed,
        )


def avatar_box(env):
    parts = connected_components(env.frame(), colors=(9, 11), min_area=1)
    if not parts:
        return None
    return (
        min(part.bbox[0] for part in parts),
        min(part.bbox[1] for part in parts),
        max(part.bbox[2] for part in parts),
        max(part.bbox[3] for part in parts),
    )


def trace_moves(env, prefix, actions):
    clone = env.clone()
    for action in prefix:
        if isinstance(action, tuple):
            clone.step(*action)
        else:
            clone.step(action)
    trace = [avatar_box(clone)]
    rewards = [clone.levels_completed]
    for action in actions:
        clone.step(action)
        trace.append(avatar_box(clone))
        rewards.append(clone.levels_completed)
        if clone.levels_completed:
            break
    print("trace", actions, "from", len(prefix), "setup", trace, rewards)


def unavailable_probe(env):
    for point in (None, (15, 3), (21, 39), (58, 58)):
        clone = env.clone()
        try:
            if point is None:
                clone.step(7)
            else:
                clone.step(7, *point)
            print("action_7", point, "accepted", frame_delta(env.frame(), clone.frame()))
        except Exception as exc:
            print("action_7", point, "rejected", type(exc).__name__)


def symbolic_bfs(env):
    centers = [(15, 3), (21, 3), (27, 3), (33, 15), (39, 15), (45, 15), (51, 15)]
    actions = [3, 4] + [(6, x, y) for x, y in centers]

    def key(node):
        return np.asarray(node.frame())[:63].tobytes()

    queue = deque([(env.clone(), [])])
    seen = {key(env)}
    max_depth = 20
    while queue:
        node, path = queue.popleft()
        if node.levels_completed:
            print("bfs_win", path, "states", len(seen))
            return
        if len(path) >= max_depth:
            continue
        for action in actions:
            child = node.clone()
            if isinstance(action, tuple):
                child.step(*action)
            else:
                child.step(action)
            child_key = key(child)
            if child_key not in seen:
                seen.add(child_key)
                queue.append((child, path + [action]))
    print("bfs_no_win", "states", len(seen), "depth", max_depth)


def aligned_clear(env):
    clone = env.clone()
    targets = [(15, 3), (21, 3), (27, 3), (33, 15),
               (39, 15), (45, 15), (51, 15)]
    current_x = 21
    path = []
    for target_x, target_y in targets:
        while current_x > target_x:
            clone.step(3)
            path.append(3)
            current_x -= 6
        while current_x < target_x:
            clone.step(4)
            path.append(4)
            current_x += 6
        clone.step(6, target_x, target_y)
        path.append((6, target_x, target_y))
        remaining = [
            (b.bbox, b.area)
            for b in connected_components(clone.frame(), colors=(14,), min_area=1)
        ]
        print(
            "aligned",
            target_x,
            "yellow",
            remaining,
            "avatar",
            avatar_box(clone),
            "counts",
            color_counts(clone.frame()),
            "reward",
            clone.levels_completed,
            "terminal",
            clone.terminal(),
        )
    print("aligned_result", clone.levels_completed, path)


def greedy_waves(env):
    clone = env.clone()
    previous_count = None
    for step in range(1, 81):
        targets = connected_components(clone.frame(), colors=(14,), min_area=4)
        if targets:
            target = targets[0]
            row, col = target.centroid
            action = (6, int(round(col)), int(round(row)))
            clone.step(*action)
        else:
            action = 6
            clone.step(action)
        after = connected_components(clone.frame(), colors=(14,), min_area=4)
        count = len(after)
        timer = color_counts(clone.frame()).get(15, 0)
        print(
            "wave_step",
            step,
            "action",
            action,
            "remaining",
            count,
            "timer",
            timer,
            "reward",
            clone.levels_completed,
            "terminal",
            clone.terminal(),
        )
        if clone.levels_completed or clone.terminal():
            break
        previous_count = count


def run(env):
    greedy_waves(env)


arena.run_program("bp35", run)
