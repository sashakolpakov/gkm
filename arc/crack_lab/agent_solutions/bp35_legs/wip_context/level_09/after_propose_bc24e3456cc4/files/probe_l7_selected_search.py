"""Replay BFS whose key observes level-7's currently selected object."""

from collections import deque
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import run_actions
from perception import connected_components
from probe_l7_frontier import BASE, L, R, avatar


def small_supports(frame):
    pixels = np.asarray(frame)
    out = []
    for y in range(1, 62):
        for x in range(13, 61):
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


def choices(node):
    frame = node.frame()
    position = avatar(frame)
    if position is None:
        return []
    ax, ay = position
    out = [L, R, (7,)]
    controls = [
        blob
        for blob in connected_components(frame, colors=(8,), min_area=3)
        if blob.bbox[1] <= 5 and blob.bbox[0] < 63
    ]
    if controls:
        blob = min(controls, key=lambda item: abs(item.centroid[0] - ay))
        y, x = blob.centroid
        out.append((6, int(round(x)), int(round(y))))
    targets = set()
    for x, y in small_supports(frame):
        if abs(x - ax) <= 13 and abs(y - ay) <= 31:
            targets.add((x, y))
    for blob in connected_components(frame, colors=(12,), min_area=6):
        y, x = blob.centroid
        x, y = int(round(x)), int(round(y))
        if abs(x - ax) <= 13 and abs(y - ay) <= 31:
            targets.add((x, y))
    out.extend((6, x, y) for x, y in sorted(targets))
    return list(dict.fromkeys(out))


def selected_key(node):
    frame = np.asarray(node.frame())[:63].tobytes()
    probe = node.clone()
    probe.step(7)
    effect = (
        bool(probe.terminal()),
        np.asarray(probe.frame())[:63].tobytes(),
    )
    return frame, effect


def search(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    run_actions(env, BASE)
    base_level = int(env.levels_completed)
    root = env.clone()
    queue = deque([()])
    seen = {selected_key(root)}
    max_states = int(os.environ.get("MAX_STATES", "500"))
    max_depth = int(os.environ.get("MAX_DEPTH", "32"))
    started = time.monotonic()
    for expanded in range(1, max_states + 1):
        if not queue:
            break
        path = queue.popleft()
        node = root.clone()
        run_actions(node, path)
        if len(path) >= max_depth or node.terminal():
            continue
        for action in choices(node):
            child = node.clone()
            child.step(*action)
            new_path = path + (action,)
            if child.levels_completed > base_level:
                route = list(BASE) + list(new_path)
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
            state = selected_key(child)
            if state in seen:
                continue
            seen.add(state)
            queue.append(new_path)
        if expanded % 25 == 0:
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
