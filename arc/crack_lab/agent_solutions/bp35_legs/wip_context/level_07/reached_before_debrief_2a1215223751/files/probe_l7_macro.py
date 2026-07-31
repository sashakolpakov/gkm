"""Room-local BFS driven by verified upward camera progress."""

from collections import deque
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import band_shift, run_actions
from perception import connected_components


LEFT = (3,)
RIGHT = (4,)
CLICK = 6

SEED = [
    RIGHT,
    RIGHT,
    RIGHT,
    (6, 39, 51),
    (6, 3, 3),
    RIGHT,
    (6, 3, 3),
    RIGHT,
    LEFT,
    LEFT,
    LEFT,
    LEFT,
    (6, 27, 57),
    (6, 3, 3),
    (6, 27, 23),
    (6, 3, 3),
    LEFT,
    LEFT,
]


def avatar_position(frame):
    ys, xs = np.where(np.asarray(frame) == 9)
    if len(xs) == 0:
        return None
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def control_action(frame):
    controls = [
        blob
        for blob in connected_components(frame, colors=(8,), min_area=3)
        if blob.bbox[1] <= 5 and blob.bbox[0] < 63
    ]
    if not controls:
        return None
    blob = controls[0]
    return CLICK, 3, int(round(blob.centroid[0]))


def support_actions(frame):
    pixels = np.asarray(frame)
    avatar = avatar_position(frame)
    if avatar is None:
        return []
    ax, ay = avatar
    targets = set()
    for y in range(1, 62):
        for x in range(1, 63):
            if (
                int(pixels[y, x]) == 12
                and abs(x - ax) <= 4
                and abs(y - ay) <= 24
                and all(
                    int(pixels[y + dy, x + dx]) == 12
                    for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1))
                )
                and all(
                    int(pixels[y + dy, x + dx]) != 12
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
            ):
                targets.add((x, y))
    for blob in connected_components(frame, colors=(12, 14), min_area=6):
        x = int(round(blob.centroid[1]))
        y = int(round(blob.centroid[0]))
        if abs(x - ax) <= 4 and abs(y - ay) <= 24:
            targets.add((x, y))
    return [(CLICK, x, y) for x, y in sorted(targets, key=lambda p: abs(p[1] - 39))]


def choices(node):
    out = [LEFT, RIGHT]
    control = control_action(node.frame())
    if control is not None:
        out.append(control)
    out.extend(support_actions(node.frame()))
    return out


def key(node, gravity):
    return np.asarray(node.frame())[:63].tobytes(), gravity


def reconstruct(root, route):
    node = root.clone()
    run_actions(node, route)
    return node


def next_rise(root, base_route, gravity, max_states=800, max_depth=14):
    start = reconstruct(root, base_route)
    queue = deque([((), gravity)])
    seen = {key(start, gravity)}
    expanded = 0
    while queue and expanded < max_states:
        path, phase = queue.popleft()
        node = reconstruct(root, base_route + list(path))
        expanded += 1
        if len(path) >= max_depth:
            continue
        for action in choices(node):
            child = node.clone()
            child.step(*action)
            new_phase = phase ^ (action[0] == CLICK and action[1] == 3)
            new_path = path + (action,)
            if child.levels_completed > root.levels_completed:
                return list(new_path), new_phase, "level", expanded
            if child.terminal() or avatar_position(child.frame()) is None:
                continue
            gain = band_shift(node.frame(), child.frame())
            if gain:
                return list(new_path), new_phase, gain, expanded
            state = key(child, new_phase)
            if state not in seen:
                seen.add(state)
                queue.append((new_path, new_phase))
    return [], gravity, 0, expanded


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    root = env.clone()
    node = reconstruct(root, SEED)
    route = list(SEED)
    gravity = 0
    print(
        "seed",
        {
            "len": len(route),
            "terminal": node.terminal(),
            "choices": choices(node),
        },
        flush=True,
    )
    for stage in range(12):
        suffix, gravity, gain, expanded = next_rise(
            root,
            route,
            gravity,
            max_states=int(os.environ.get("PROBE_STATES", "300")),
            max_depth=int(os.environ.get("PROBE_DEPTH", "12")),
        )
        print(
            "stage",
            stage,
            {"gain": gain, "expanded": expanded, "suffix": suffix},
            flush=True,
        )
        if not suffix:
            break
        route.extend(suffix)
        node = reconstruct(root, route)
        if node.levels_completed > env.levels_completed:
            break
    print(
        "result",
        {
            "len": len(route),
            "route": route,
            "level_delta": int(node.levels_completed - env.levels_completed),
            "terminal": bool(node.terminal()),
        },
    )


arena.run_program("bp35", probe)
