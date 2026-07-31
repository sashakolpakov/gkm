"""Direct-clone BFS over level-7 gravity, movement, and nearby supports."""

from collections import deque
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components


LEFT = (3,)
RIGHT = (4,)
UP = (7, 0, 0)
OPENING = [
    RIGHT,
    RIGHT,
    RIGHT,
    (6, 39, 51),
    (6, 3, 3),
    RIGHT,
    (6, 3, 3),
    RIGHT,
]


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 9)
    if not len(xs):
        return None
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def small_supports(frame):
    pixels = np.asarray(frame)
    out = []
    for y in range(1, 62):
        for x in range(1, 63):
            if (
                int(pixels[y, x]) == 12
                and all(
                    int(pixels[y + dy, x + dx]) == 12
                    for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1))
                )
                and all(
                    int(pixels[y + dy, x + dx]) != 12
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
            ):
                out.append((x, y))
    return out


def choices(frame):
    position = avatar(frame)
    if position is None:
        return []
    ax, ay = position
    out = [LEFT, RIGHT, UP]
    controls = [
        blob
        for blob in connected_components(frame, colors=(8,), min_area=3)
        if blob.bbox[1] <= 5 and blob.bbox[0] < 63
    ]
    if controls:
        y, x = controls[0].centroid
        out.append((6, int(round(x)), int(round(y))))
    for x, y in small_supports(frame):
        if abs(x - ax) <= 7 and abs(y - ay) <= 30:
            out.append((6, x, y))
    for blob in connected_components(frame, colors=(12,), min_area=6):
        y, x = blob.centroid
        x, y = int(round(x)), int(round(y))
        if abs(x - ax) <= 7 and abs(y - ay) <= 30:
            out.append((7, x, y))
    return list(dict.fromkeys(out))


def key(frame):
    return np.asarray(frame)[:63].tobytes()


def search(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    base_level = int(env.levels_completed)
    for action in OPENING if os.environ.get("NO_OPENING") != "1" else ():
        env.step(*action)
    root = env.clone()
    queue = deque([(root, ())])
    seen = {key(root.frame())}
    started = time.monotonic()
    max_states = int(os.environ.get("MAX_STATES", "3000"))
    max_depth = int(os.environ.get("MAX_DEPTH", "70"))
    for expanded in range(1, max_states + 1):
        if not queue:
            break
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for action in choices(node.frame()):
            child = node.clone()
            child.step(*action)
            new_path = path + (action,)
            if child.levels_completed > base_level:
                route = list(OPENING) + list(new_path)
                print(
                    {
                        "found": True,
                        "expanded": expanded,
                        "seconds": round(time.monotonic() - started, 3),
                        "route_len": len(route),
                        "route": route,
                    },
                    flush=True,
                )
                return
            if child.terminal() or avatar(child.frame()) is None:
                continue
            state = key(child.frame())
            if state in seen:
                continue
            seen.add(state)
            queue.append((child, new_path))
        if expanded % 100 == 0:
            print(
                {
                    "expanded": expanded,
                    "queued": len(queue),
                    "depth": len(path),
                    "seconds": round(time.monotonic() - started, 3),
                },
                flush=True,
            )
    print(
        {
            "found": False,
            "expanded": expanded,
            "queued": len(queue),
            "seen": len(seen),
            "seconds": round(time.monotonic() - started, 3),
        },
        flush=True,
    )


arena.run_program("bp35", search)
