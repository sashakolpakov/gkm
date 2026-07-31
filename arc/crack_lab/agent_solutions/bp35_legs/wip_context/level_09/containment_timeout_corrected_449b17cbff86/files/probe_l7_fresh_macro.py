"""Fresh-replay local BFS, avoiding recursively nested Arena clones."""

from collections import deque
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import band_shift
from perception import connected_components


LEFT = (3,)
RIGHT = (4,)
CLICK = 6

with open("checkpoint.json") as stream:
    PREFIX = json.load(stream)["final_path"]

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
                and abs(x - ax) <= 7
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
        if abs(x - ax) <= 7 and abs(y - ay) <= 24:
            targets.add((x, y))
    return [(CLICK, x, y) for x, y in sorted(targets)]


def choices(frame):
    out = [LEFT, RIGHT]
    control = control_action(frame)
    if control is not None:
        out.append(control)
    out.extend(support_actions(frame))
    return out


def observation_key(frame, gravity):
    return np.asarray(frame)[:63].tobytes(), gravity


def evaluate(path, gravity):
    result = {}

    def callback(env):
        for action in PREFIX:
            env.step(action)
        for action in path:
            if env.terminal():
                break
            env.step(*action)
        if env.terminal():
            result["children"] = []
            return
        before = env.frame()
        base_level = int(env.levels_completed)
        children = []
        for action in choices(before):
            child = env.clone()
            child.step(*action)
            phase = gravity ^ (action[0] == CLICK and action[1] == 3)
            children.append(
                {
                    "action": action,
                    "phase": phase,
                    "key": observation_key(child.frame(), phase),
                    "gain": band_shift(before, child.frame()),
                    "level": int(child.levels_completed) - base_level,
                    "terminal": bool(child.terminal()),
                    "avatar": avatar_position(child.frame()),
                }
            )
        result["children"] = children

    arena.run_program("bp35", callback)
    return result.get("children", [])


def search(max_states=500, max_depth=12):
    queue = deque([(list(SEED), (), 0)])
    seen = set()
    started = time.monotonic()
    for expanded in range(1, max_states + 1):
        if not queue:
            break
        route, suffix, gravity = queue.popleft()
        for child in evaluate(route, gravity):
            action = child["action"]
            new_suffix = suffix + (action,)
            if child["level"] > 0 or child["gain"] > 0:
                print(
                    {
                        "expanded": expanded,
                        "seconds": round(time.monotonic() - started, 3),
                        "gain": child["gain"],
                        "level": child["level"],
                        "suffix": new_suffix,
                    }
                )
                return list(new_suffix)
            if (
                child["terminal"]
                or child["avatar"] is None
                or len(new_suffix) >= max_depth
                or child["key"] in seen
            ):
                continue
            seen.add(child["key"])
            queue.append((route + [action], new_suffix, child["phase"]))
        if expanded % 25 == 0:
            print(
                {
                    "expanded": expanded,
                    "queued": len(queue),
                    "seconds": round(time.monotonic() - started, 3),
                },
                flush=True,
            )
    print({"expanded": expanded, "queued": len(queue), "found": False})
    return []


search()
