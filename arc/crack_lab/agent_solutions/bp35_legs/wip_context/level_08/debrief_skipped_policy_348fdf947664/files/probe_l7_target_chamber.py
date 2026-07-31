"""Small selected-aware search from the first safe drop beside the target."""

from collections import deque
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components, frame_delta


L, R = (3,), (4,)
ENTRY = [
    R, R, R,
    (6, 39, 51),
    (6, 3, 3),
    R,
    (6, 3, 3),
    (7,),
    (6, 51, 51),
    (7,),
    R,
]


def center(frame, color):
    blobs = [
        blob for blob in connected_components(frame, colors=(color,), min_area=2)
        if blob.bbox[0] < 63
    ]
    if not blobs:
        return None
    y, x = blobs[0].centroid
    return round(x), round(y)


def distance(frame):
    avatar, target = center(frame, 9), center(frame, 7)
    if avatar is None or target is None:
        return 99
    return round((abs(avatar[0] - target[0]) + abs(avatar[1] - target[1])) / 6)


def choices(node):
    frame = node.frame()
    avatar = center(frame, 9)
    if avatar is None:
        return []
    ax, ay = avatar
    out = [L, R, (7,)]
    controls = [
        blob for blob in connected_components(frame, colors=(8,), min_area=2)
        if blob.bbox[0] < 63 and blob.bbox[1] <= 5
    ]
    if controls:
        blob = min(controls, key=lambda item: abs(item.centroid[0] - ay))
        y, x = blob.centroid
        out.append((6, round(x), round(y)))
    for color in (0, 7, 12, 15):
        for blob in connected_components(frame, colors=(color,), min_area=2):
            if blob.bbox[0] >= 63:
                continue
            y, x = blob.centroid
            if abs(x - ax) <= 18 and abs(y - ay) <= 30:
                out.append((6, round(x), round(y)))
    return list(dict.fromkeys(out))


def selected_key(node):
    frame = np.asarray(node.frame())[:63].tobytes()
    probe = node.clone()
    probe.step(7)
    return frame, bool(probe.terminal()), np.asarray(probe.frame())[:63].tobytes()


def summary(node):
    frame = node.frame()
    return center(frame, 9), center(frame, 7), distance(frame), choices(node)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for index, action in enumerate(ENTRY, 1):
        env.step(*action)
        pixels = np.asarray(env.frame())
        print(
            "ENTRY_STEP", index, action, env.terminal(),
            int(np.count_nonzero(pixels == 9)),
            int(np.count_nonzero(pixels == 11)),
            center(env.frame(), 9),
            center(env.frame(), 7),
        )
        if env.terminal():
            return

    base_level = int(env.levels_completed)
    root = env.clone()
    print("ROOT", summary(root))
    before = np.asarray(root.frame()).copy()
    for action in choices(root):
        child = root.clone()
        child.step(*action)
        delta = frame_delta(before[:63], child.frame()[:63])
        print(
            "ONE", action, child.levels_completed, child.terminal(),
            None if child.terminal() else summary(child),
            (delta["count"], delta["bbox"]),
        )

    queue = deque([(root, ())])
    seen = {selected_key(root)}
    best = (distance(root.frame()), ())
    expanded = 0
    while queue and expanded < 160:
        node, path = queue.popleft()
        if len(path) >= 12:
            continue
        for action in choices(node):
            child = node.clone()
            child.step(*action)
            expanded += 1
            child_path = (*path, action)
            if child.levels_completed > base_level:
                print("WIN", child_path, expanded)
                return
            if child.terminal() or center(child.frame(), 9) is None:
                continue
            progress = distance(child.frame())
            if progress < best[0]:
                best = (progress, child_path)
                print("PROGRESS", best, summary(child))
            key = selected_key(child)
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path))
            if expanded >= 160:
                break
    print("SEARCH", expanded, len(seen), len(queue), best)


levels, path, error = arena.run_program("bp35", probe)
print("RESULT", levels, len(path), error)
